// MIT License, Copyright (c) 2016 Mariela Perignon.
// Port of pyDeltaRCM/water_tools.py; see LICENSE for the full license.
use std::collections::HashSet;

use numpy::{
    ndarray::Array2, IntoPyArray, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3,
    PyUntypedArrayMethods,
};
use pyo3::{exceptions::PyValueError, prelude::*, types::PyDict};
use rand::{Rng, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;

#[pyclass(frozen)]
struct WaterConfig {
    stepmax: usize,
    l0: usize,
    ctr: usize,
    columns: usize,
    nsmooth: usize,
    dx: f64,
    qp: f64,
    u0: f64,
    h0: f64,
    sea_level: f64,
    slope: f64,
    csmooth: f64,
}

#[pymethods]
impl WaterConfig {
    #[new]
    #[allow(clippy::too_many_arguments)]
    fn new(
        stepmax: usize,
        l0: usize,
        ctr: usize,
        columns: usize,
        nsmooth: usize,
        dx: f64,
        qp: f64,
        u0: f64,
        h0: f64,
        sea_level: f64,
        slope: f64,
        csmooth: f64,
    ) -> PyResult<Self> {
        if columns == 0
            || dx <= 0.0
            || h0 <= 0.0
            || ![dx, qp, u0, h0, sea_level, slope, csmooth]
                .iter()
                .all(|x| x.is_finite())
        {
            return Err(PyValueError::new_err("invalid water-phase configuration"));
        }
        Ok(Self {
            stepmax,
            l0,
            ctr,
            columns,
            nsmooth,
            dx,
            qp,
            u0,
            h0,
            sea_level,
            slope,
            csmooth,
        })
    }
}

struct Fields<'a> {
    rows: usize,
    cols: usize,
    cells: &'a [i64],
    stage: &'a [f32],
    eta: &'a [f32],
    depth: &'a [f32],
    uw: &'a [f32],
    ux: &'a [f32],
    uy: &'a [f32],
}

struct Routing {
    qx: Vec<f32>,
    qy: Vec<f32>,
    qw: Vec<f32>,
    paths: Vec<Vec<i64>>,
    flags: Vec<i64>,
    columns: usize,
}

fn choose_direction(weights: &[f64], uniform: f64) -> Result<usize, String> {
    let total = weights.iter().sum::<f64>();
    // Mirror the reference random_pick guard: a nonpositive (or NaN) total
    // would silently pick direction 0 in the inverse-CDF scan.
    if total <= 0.0 || total.is_nan() {
        return Err("probability weights must have a positive total".into());
    }
    let target = uniform * total;
    let mut cumulative = 0.0;
    for (direction, weight) in weights.iter().enumerate() {
        cumulative += weight;
        if cumulative >= target {
            return Ok(direction);
        }
    }
    Ok(8)
}

fn flux(
    qx: &mut [f32],
    qy: &mut [f32],
    qw: &mut [f32],
    index: usize,
    direction: usize,
    amount: f64,
) {
    if direction == 4 {
        return;
    }
    let row = (direction / 3) as i64 - 1;
    let col = (direction % 3) as i64 - 1;
    let distance = ((row * row + col * col) as f64).sqrt();
    qx[index] = (f64::from(qx[index]) + row as f64 / distance) as f32;
    qy[index] = (f64::from(qy[index]) + col as f64 / distance) as f32;
    qw[index] = (f64::from(qw[index]) + amount) as f32;
}

fn relocate(index: usize, fields: &Fields<'_>, config: &WaterConfig) -> usize {
    let row = index / fields.cols;
    let col = index % fields.cols;
    let fx = row as f64 - 1.0;
    let fy = col as f64 - config.ctr as f64;
    let magnitude = fx.hypot(fy);
    // A zero transport vector means the loop was detected at the origin
    // (1, CTR); the reference relocates one row downstream of the loop
    // cell instead of stranding the parcel at the vector origin.
    let (dr, dc) = if magnitude == 0.0 {
        (1, 0)
    } else {
        (
            (fx / magnitude * 5.0).round_ties_even() as i64,
            (fy / magnitude * 5.0).round_ties_even() as i64,
        )
    };
    let row = (row as i64 + dr).clamp(config.l0 as i64, fields.rows as i64 - 2);
    let col = (col as i64 + dc).clamp(1, fields.cols as i64 - 2);
    row as usize * fields.cols + col as usize
}

// NumPy flat indexing wraps negative inlet exits onto the opposite boundary.
// Preserve the signed path entry (surface accumulation ignores nonpositives).
fn wrap_index(index: i64, size: usize) -> Result<usize, String> {
    if index < -(size as i64) || index as i128 >= size as i128 {
        return Err("water parcel index outside NumPy flat-index range".into());
    }
    Ok(if index < 0 {
        (size as i64 + index) as usize
    } else {
        index as usize
    })
}

fn route(
    fields: &Fields<'_>,
    weights: &[f64],
    starts: &[i64],
    config: &WaterConfig,
    seed: u64,
) -> Result<Routing, String> {
    let size = fields.cells.len();
    let count = starts.len();
    let mut out = Routing {
        qx: vec![0.0; size],
        qy: vec![0.0; size],
        qw: vec![0.0; size],
        paths: starts.iter().map(|&s| vec![s]).collect(),
        flags: vec![1; count],
        columns: config.columns,
    };
    let amount = config.qp / config.dx / 2.0;
    let mut inlet_counts = vec![0_usize; size];
    for &start in starts {
        inlet_counts[start as usize] += 1;
    }
    for (index, &count) in inlet_counts.iter().enumerate() {
        out.qx[index] = count as f32;
        // Match Python's unique/count reduction, including its single f32 cast.
        out.qw[index] = (count as f64 * config.qp / config.dx / 2.0) as f32;
    }
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
    let mut current = starts.to_vec();
    let mut next = vec![0; count];
    let mut directions = vec![4; count];
    let mut seen: Vec<HashSet<usize>> = starts
        .iter()
        .map(|&s| {
            let mut set = HashSet::new();
            if s > 0 && s as usize >= config.l0 * fields.cols {
                set.insert(s as usize);
            }
            set
        })
        .collect();
    for step in 1..=config.stepmax {
        if current.iter().map(|&index| i128::from(index)).sum::<i128>() <= 0 {
            break;
        }
        if step >= out.columns {
            out.columns += (config.stepmax / 4).max(1);
        }
        for p in 0..count {
            let index = wrap_index(current[p], size)?;
            directions[p] = if current[p] == 0 {
                4
            } else {
                choose_direction(&weights[index * 9..index * 9 + 9], rng.random::<f64>())?
            };
            let d = directions[p];
            next[p] = if d == 4 {
                0
            } else {
                current[p] + (d as i64 / 3 - 1) * fields.cols as i64 + d as i64 % 3 - 1
            };
            if next[p] < -(size as i64) || next[p] as i128 >= size as i128 {
                return Err(format!(
                    "water parcel stepped outside the grid: step={step}, parcel={p}, current={}, direction={d}, next={}; check boundary weights",
                    current[p], next[p]
                ));
            }
        }
        // Preserve the reference's accumulation order: all exits, then all entries.
        for p in 0..count {
            flux(
                &mut out.qx,
                &mut out.qy,
                &mut out.qw,
                wrap_index(current[p], size)?,
                directions[p],
                amount,
            );
        }
        for p in 0..count {
            flux(
                &mut out.qx,
                &mut out.qy,
                &mut out.qw,
                wrap_index(next[p], size)?,
                directions[p],
                amount,
            );
        }
        for p in 0..count {
            let mut raw_index = next[p];
            let mut index = wrap_index(raw_index, size)?;
            let mut looped = false;
            if step > config.l0 && raw_index > 0 && seen[p].contains(&index) {
                looped = (fields.stage[index] - config.sea_level as f32).abs() >= 0.1_f32;
                index = relocate(index, fields, config);
                raw_index = index as i64;
                if looped {
                    out.flags[p] = 0;
                }
            }
            let boundary = fields.cells[index] == -1;
            let saved = if boundary && looped { 0 } else { raw_index };
            out.paths[p].push(saved);
            if saved > 0 && saved as usize >= config.l0 * fields.cols {
                seen[p].insert(saved as usize);
            }
            current[p] = if boundary { 0 } else { raw_index };
            if boundary {
                flux(
                    &mut out.qx,
                    &mut out.qy,
                    &mut out.qw,
                    index,
                    directions[p],
                    amount,
                );
            }
        }
    }
    Ok(out)
}

fn accumulate(
    fields: &Fields<'_>,
    paths: &[Vec<i64>],
    flags: &[i64],
    config: &WaterConfig,
) -> (Vec<f64>, Vec<f64>) {
    let size = fields.cells.len();
    let mut visits = vec![0.0; size];
    let mut sums = vec![0.0; size];
    let mut heights = vec![0.0; size];
    for (path, &flag) in paths.iter().zip(flags) {
        let path: Vec<usize> = path
            .iter()
            .filter(|&&i| i > 0)
            .map(|&i| i as usize)
            .collect();
        let Some(&last) = path.last() else {
            continue;
        };
        if flag != 1 || fields.cells[last] != -1 {
            continue;
        }
        heights.fill(0.0);
        heights[last] = config.sea_level;
        let mut in_ocean = true;
        let mut dh = 0.0;
        for pair in path.windows(2).rev() {
            let (index, next) = (pair[0], pair[1]);
            if index != next {
                if in_ocean {
                    if fields.uw[index] as f64 > config.u0 * 0.5
                        || (fields.depth[index] as f64) < 0.1 * config.h0
                    {
                        in_ocean = false;
                    }
                } else if fields.uw[index] == 0.0 {
                    dh = 0.0;
                } else {
                    let dr = (next / fields.cols) as i64 - (index / fields.cols) as i64;
                    let dc = (next % fields.cols) as i64 - (index % fields.cols) as i64;
                    dh = config.slope
                        * (fields.ux[index] as f64 * dr as f64 * config.dx
                            + fields.uy[index] as f64 * dc as f64 * config.dx)
                        / fields.uw[index] as f64;
                }
            }
            heights[index] = heights[next] + dh;
            visits[index] += 1.0;
            sums[index] += heights[index];
        }
    }
    (visits, sums)
}

fn smooth(input: &[f32], fields: &Fields<'_>, config: &WaterConfig) -> Vec<f32> {
    let mut result = input.to_vec();
    for _ in 0..config.nsmooth {
        let old = result.clone();
        for row in 0..fields.rows {
            for col in 0..fields.cols {
                let index = row * fields.cols + col;
                if fields.cells[index] == -1 {
                    continue;
                }
                let mut sum = 0.0_f32;
                let mut count = 0;
                for direction in 0..9 {
                    if direction == 4 {
                        continue;
                    }
                    let r =
                        (row as i64 + direction / 3 - 1).clamp(0, fields.rows as i64 - 1) as usize;
                    let c =
                        (col as i64 + direction % 3 - 1).clamp(0, fields.cols as i64 - 1) as usize;
                    let neighbor = r * fields.cols + c;
                    if fields.cells[neighbor] != -2 {
                        sum += old[neighbor];
                        count += 1;
                    }
                }
                if count > 0 {
                    result[index] = (config.csmooth * old[index] as f64
                        + (1.0 - config.csmooth) * sum as f64 / count as f64)
                        as f32;
                }
            }
        }
    }
    result
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn water_phase<'py>(
    py: Python<'py>,
    weights: PyReadonlyArray3<'py, f64>,
    starts: PyReadonlyArray1<'py, i64>,
    cells: PyReadonlyArray2<'py, i64>,
    stage: PyReadonlyArray2<'py, f32>,
    eta: PyReadonlyArray2<'py, f32>,
    depth: PyReadonlyArray2<'py, f32>,
    uw: PyReadonlyArray2<'py, f32>,
    ux: PyReadonlyArray2<'py, f32>,
    uy: PyReadonlyArray2<'py, f32>,
    config: &WaterConfig,
    seed: u64,
) -> PyResult<Bound<'py, PyDict>> {
    let shape = cells.shape();
    let (rows, cols) = (shape[0], shape[1]);
    if rows < 3
        || cols < 3
        || config.l0 > rows - 2
        || config.ctr >= cols
        || weights.shape() != [rows, cols, 9]
        || [
            stage.shape(),
            eta.shape(),
            depth.shape(),
            uw.shape(),
            ux.shape(),
            uy.shape(),
        ]
        .iter()
        .any(|s| *s != shape)
    {
        return Err(PyValueError::new_err(
            "incompatible water-phase grid shapes or inlet geometry",
        ));
    }
    let starts = starts.as_slice()?;
    if starts.iter().any(|&i| i < 0 || i as usize >= rows * cols) {
        return Err(PyValueError::new_err(
            "start index outside water-phase grid",
        ));
    }
    let weights = weights.as_slice()?;
    if weights.iter().any(|&w| !w.is_finite() || w < 0.0) {
        return Err(PyValueError::new_err(
            "water weights must be finite and nonnegative",
        ));
    }
    let fields = Fields {
        rows,
        cols,
        cells: cells.as_slice()?,
        stage: stage.as_slice()?,
        eta: eta.as_slice()?,
        depth: depth.as_slice()?,
        uw: uw.as_slice()?,
        ux: ux.as_slice()?,
        uy: uy.as_slice()?,
    };
    let routed = route(&fields, weights, starts, config, seed).map_err(PyValueError::new_err)?;
    let (visits, sums) = accumulate(&fields, &routed.paths, &routed.flags, config);
    let hnew: Vec<f32> = (0..rows * cols)
        .map(|i| {
            let value = if visits[i] > 0.0 {
                (sums[i] / visits[i]) as f32
            } else {
                fields.eta[i] + fields.depth[i]
            };
            value.max(config.sea_level as f32).max(fields.eta[i])
        })
        .collect();
    let hsmth = smooth(&hnew, &fields, config);
    let result = PyDict::new(py);
    for (name, data) in [
        ("qxn", routed.qx),
        ("qyn", routed.qy),
        ("qwn", routed.qw),
        ("Hnew", hnew),
        ("Hsmth", hsmth),
    ] {
        result.set_item(
            name,
            Array2::from_shape_vec((rows, cols), data)
                .map_err(|e| PyValueError::new_err(e.to_string()))?
                .into_pyarray(py),
        )?;
    }
    for (name, data) in [("sfc_visit", visits), ("sfc_sum", sums)] {
        result.set_item(
            name,
            Array2::from_shape_vec((rows, cols), data)
                .map_err(|e| PyValueError::new_err(e.to_string()))?
                .into_pyarray(py),
        )?;
    }
    let mut paths = Array2::<i64>::zeros((starts.len(), routed.columns));
    for (p, path) in routed.paths.iter().enumerate() {
        for (s, &i) in path.iter().enumerate() {
            paths[[p, s]] = i;
        }
    }
    result.set_item("free_surf_walk_inds", paths.into_pyarray(py))?;
    result.set_item("free_surf_flag", routed.flags.into_pyarray(py))?;
    Ok(result)
}

#[pymodule]
fn _native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<WaterConfig>()?;
    module.add_function(wrap_pyfunction!(water_phase, module)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const ROWS: usize = 10;
    const COLS: usize = 10;
    const CTR: usize = 4;

    fn test_config(stepmax: usize) -> WaterConfig {
        WaterConfig {
            stepmax,
            l0: 1,
            ctr: CTR,
            columns: 8,
            nsmooth: 1,
            dx: 10.0,
            qp: 10.0,
            u0: 1.0,
            h0: 1.0,
            sea_level: 0.0,
            slope: 0.001,
            csmooth: 0.1,
        }
    }

    fn zeros() -> Vec<f32> {
        vec![0.0; ROWS * COLS]
    }

    #[test]
    fn relocate_zero_transport_moves_one_row_downstream() {
        let cells = vec![1i64; ROWS * COLS];
        let zeros = zeros();
        let fields = Fields {
            rows: ROWS,
            cols: COLS,
            cells: &cells,
            stage: &zeros,
            eta: &zeros,
            depth: &zeros,
            uw: &zeros,
            ux: &zeros,
            uy: &zeros,
        };
        let config = test_config(5);
        // Loop detected at the transport origin (1, CTR): Fx == Fy == 0.
        let origin = COLS + CTR;
        assert_eq!(relocate(origin, &fields, &config), 2 * COLS + CTR);
    }

    #[test]
    fn choose_direction_rejects_nonpositive_total() {
        // All-zero weights would silently pick direction 0 without a guard.
        assert!(choose_direction(&[0.0; 9], 0.5).is_err());
        assert!(choose_direction(&[f64::NAN; 9], 0.5).is_err());
        // Sanity: positive weights still pick (uniform 9-cell pick, half the
        // cumulative mass crossed at direction 4).
        assert_eq!(choose_direction(&[1.0; 9], 0.5).unwrap(), 4);
    }
}
