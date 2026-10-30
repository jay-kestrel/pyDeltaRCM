// MIT License, Copyright (c) 2016 Mariela Perignon; see LICENSE.
// Phase-level port of pyDeltaRCM/sed_tools.py, including deposit-first mud routing.
use numpy::{ndarray::Array2, IntoPyArray, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::{exceptions::PyValueError, prelude::*, types::PyDict};
use rand::{Rng, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;

#[derive(FromPyObject)]
#[pyo3(from_item_all)]
struct Config {
    dt: f64,
    dx: f64,
    volume: f64,
    u_max: f64,
    dry_depth: f64,
    beta: f64,
    theta: f64,
    stepmax: usize,
    force_deposit: bool,
    sand: bool,
    qs0: f64,
    u0: f64,
    bedload: f64,
    erosion_velocity: f64,
    deposition_velocity: f64,
    settling: f64,
}

impl Config {
    fn validate(&self) -> PyResult<()> {
        let values = [
            self.dt,
            self.dx,
            self.volume,
            self.u_max,
            self.dry_depth,
            self.beta,
            self.theta,
            self.qs0,
            self.u0,
            self.bedload,
            self.erosion_velocity,
            self.deposition_velocity,
            self.settling,
        ];
        if !values.iter().all(|v| v.is_finite() && *v >= 0.0)
            || self.dt == 0.0
            || self.dx == 0.0
            || self.stepmax == 0
            || self.erosion_velocity == 0.0
            || (self.sand && self.u0 == 0.0)
        {
            return Err(PyValueError::new_err(
                "invalid sediment-phase configuration",
            ));
        }
        Ok(())
    }
}

struct Fields {
    rows: usize,
    cols: usize,
    eta: Vec<f32>,
    stage: Vec<f32>,
    depth: Vec<f32>,
    padded_depth: Vec<f32>,
    cells: Vec<i64>,
    uw: Vec<f32>,
    ux: Vec<f32>,
    uy: Vec<f32>,
    qw: Vec<f32>,
    qx: Vec<f32>,
    qy: Vec<f32>,
    qs: Vec<f32>,
    deposited: Vec<f32>,
    erosion_modifier: Vec<f32>,
    sediment_modifier: Vec<f32>,
    stability_modifier: Vec<f32>,
    ivec: Vec<f32>,
    jvec: Vec<f32>,
    distances: Vec<f32>,
}

fn array2(dict: &Bound<'_, PyDict>, name: &str, shape: (usize, usize)) -> PyResult<Vec<f32>> {
    let item = dict
        .get_item(name)?
        .ok_or_else(|| PyValueError::new_err(format!("missing {name}")))?;
    let array: PyReadonlyArray2<'_, f32> = item.extract()?;
    if array.as_array().dim() != shape {
        return Err(PyValueError::new_err(format!(
            "incompatible sediment grid shape: {name}"
        )));
    }
    let values = array.as_slice()?.to_vec();
    if !values.iter().all(|v| v.is_finite()) {
        return Err(PyValueError::new_err(format!(
            "nonfinite sediment field: {name}"
        )));
    }
    Ok(values)
}

fn array1(dict: &Bound<'_, PyDict>, name: &str) -> PyResult<Vec<f32>> {
    let item = dict
        .get_item(name)?
        .ok_or_else(|| PyValueError::new_err(format!("missing {name}")))?;
    let array: PyReadonlyArray1<'_, f32> = item.extract()?;
    let values = array.as_slice()?.to_vec();
    if values.len() != 9 || !values.iter().all(|v| v.is_finite()) {
        return Err(PyValueError::new_err(format!(
            "invalid sediment walk geometry: {name}"
        )));
    }
    Ok(values)
}

fn pad_depth(depth: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    (0..(rows + 2) * (cols + 2))
        .map(|i| {
            let row = (i / (cols + 2)).saturating_sub(1).min(rows - 1);
            let col = (i % (cols + 2)).saturating_sub(1).min(cols - 1);
            depth[row * cols + col]
        })
        .collect()
}

impl Fields {
    fn read(dict: &Bound<'_, PyDict>, config: &Config) -> PyResult<Self> {
        let item = dict
            .get_item("cell_type")?
            .ok_or_else(|| PyValueError::new_err("missing cell_type"))?;
        let cells: PyReadonlyArray2<'_, i64> = item.extract()?;
        let shape = cells.as_array().dim();
        let (rows, cols) = shape;
        if rows == 0 || cols == 0 {
            return Err(PyValueError::new_err("empty sediment grid"));
        }
        let depth = array2(dict, "depth", shape)?;
        let out = Self {
            rows,
            cols,
            padded_depth: pad_depth(&depth, rows, cols),
            depth,
            cells: cells.as_slice()?.to_vec(),
            eta: array2(dict, "eta", shape)?,
            stage: array2(dict, "stage", shape)?,
            uw: array2(dict, "uw", shape)?,
            ux: array2(dict, "ux", shape)?,
            uy: array2(dict, "uy", shape)?,
            qw: array2(dict, "qw", shape)?,
            qx: array2(dict, "qx", shape)?,
            qy: array2(dict, "qy", shape)?,
            qs: if config.sand {
                array2(dict, "qs", shape)?
            } else {
                vec![0.0; rows * cols]
            },
            deposited: array2(
                dict,
                if config.sand {
                    "Vp_dep_sand"
                } else {
                    "Vp_dep_mud"
                },
                shape,
            )?,
            erosion_modifier: array2(dict, "mod_erosion", shape)?,
            sediment_modifier: array2(dict, "mod_sed_weight", shape)?,
            stability_modifier: array2(dict, "mod_stable_weight", shape)?,
            ivec: array1(dict, "ivec_flat")?,
            jvec: array1(dict, "jvec_flat")?,
            distances: array1(dict, "distances_flat")?,
        };
        if out.distances.iter().any(|&d| d <= 0.0) {
            return Err(PyValueError::new_err("sediment distances must be positive"));
        }
        Ok(out)
    }

    fn weights(&self, index: usize, config: &Config) -> Result<[f64; 9], String> {
        let row = index / self.cols;
        let col = index % self.cols;
        let mut weights = [0.0; 9];
        let mut walls = [false; 9];
        for d in 0..9 {
            let r = (row + d / 3).saturating_sub(1).min(self.rows - 1);
            let c = (col + d % 3).saturating_sub(1).min(self.cols - 1);
            let n = r * self.cols + c;
            walls[d] = self.cells[n] == -2;
            let depth = self.padded_depth[(row + d / 3) * (self.cols + 2) + col + d % 3];
            let inertia = ((self.qx[index] * self.jvec[d] + self.qy[index] * self.ivec[d])
                / self.distances[d])
                .max(0.0);
            let weight = (depth * self.sediment_modifier[n]).powf(config.theta as f32) * inertia;
            if d != 4 && !walls[d] && f64::from(depth) > config.dry_depth {
                if !weight.is_finite() || weight < 0.0 {
                    return Err("invalid sediment routing weight".into());
                }
                weights[d] = f64::from(weight);
            }
        }
        if weights.iter().sum::<f64>() == 0.0 {
            for d in 0..9 {
                // Current reference excludes ghost-row escapes in its dry-cell fallback.
                if d != 4 && !walls[d] && !(row == 0 && d < 3) {
                    weights[d] = f64::from(1.0 / self.distances[d]);
                }
            }
        }
        if weights.iter().sum::<f64>() <= 0.0 {
            return Err("No weights encountered in sediment weighting.Please report error.".into());
        }
        Ok(weights)
    }

    fn next_cell(&self, index: usize, direction: usize) -> Result<usize, String> {
        let row = index as isize / self.cols as isize + direction as isize / 3 - 1;
        let col = index as isize % self.cols as isize + direction as isize % 3 - 1;
        if row < 0 || col < 0 || row >= self.rows as isize || col >= self.cols as isize {
            return Err("sediment parcel stepped outside grid; check boundary weights".into());
        }
        Ok(row as usize * self.cols + col as usize)
    }

    fn change(&self, index: usize, residual: f32, config: &Config) -> f64 {
        let velocity = f64::from(self.uw[index]);
        let capacity =
            config.qs0 * config.bedload / config.u0.powf(config.beta) * velocity.powf(config.beta);
        let deposit = if config.sand {
            f64::from(self.qs[index]) > capacity
        } else {
            velocity < config.deposition_velocity
        };
        let depth = f64::from(self.stage[index] - self.eta[index]);
        if deposit {
            let volume = if config.sand {
                f64::from(residual)
            } else {
                config.settling
                    * f64::from(residual)
                    * (config.deposition_velocity.powf(config.beta) - velocity.powf(config.beta))
                    / config.deposition_velocity.powf(config.beta)
            };
            limit_change(
                volume,
                depth,
                config.dx,
                false,
                f64::from(self.stability_modifier[index]),
            )
        } else if velocity > config.erosion_velocity
            && (!config.sand || f64::from(self.qs[index]) < capacity)
        {
            let volume = f64::from(self.erosion_modifier[index])
                * config.volume
                * (velocity.powf(config.beta) - config.erosion_velocity.powf(config.beta))
                / config.erosion_velocity.powf(config.beta);
            -limit_change(
                volume,
                depth,
                config.dx,
                true,
                f64::from(self.stability_modifier[index]),
            )
        } else {
            0.0
        }
    }

    fn update(&mut self, index: usize, change: f64, config: &Config) {
        let eta = f64::from(self.eta[index]) + change / (config.dx * config.dx);
        let depth = (f64::from(self.stage[index]) - eta).max(0.0);
        let qw = f64::from(self.qw[index]);
        let velocity = if depth > 0.0 {
            config.u_max.min(qw / depth)
        } else {
            f64::from(self.uw[index])
        };
        self.eta[index] = eta as f32;
        self.depth[index] = depth as f32;
        self.padded_depth[(index / self.cols + 1) * (self.cols + 2) + index % self.cols + 1] =
            depth as f32;
        self.uw[index] = velocity as f32;
        self.ux[index] = if qw > 0.0 {
            (velocity * f64::from(self.qx[index]) / qw) as f32
        } else {
            0.0
        };
        self.uy[index] = if qw > 0.0 {
            (velocity * f64::from(self.qy[index]) / qw) as f32
        } else {
            0.0
        };
    }
}

fn limit_change(volume: f64, depth: f64, dx: f64, erosion: bool, modifier: f64) -> f64 {
    if depth < 0.0 && !erosion {
        0.0
    } else {
        volume.min(modifier * depth.abs() / 4.0 * dx * dx)
    }
}

#[derive(Default)]
struct Volumes {
    residual: f32,
    exported: f32,
    lost: f32,
}

fn route(
    fields: &mut Fields,
    starts: &[i64],
    config: &Config,
    seed: u64,
) -> Result<Volumes, String> {
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
    let mut volumes = Volumes::default();
    for &start in starts {
        let mut index = start as usize;
        volumes.residual = config.volume as f32;
        if config.sand {
            fields.qs[index] = (f64::from(fields.qs[index])
                + f64::from(volumes.residual) / 2.0 / config.dt / config.dx)
                as f32;
        }
        for step in 1..=config.stepmax {
            let weights = fields.weights(index, config)?;
            let direction = super::choose_direction(&weights, rng.random::<f64>());
            let next = fields.next_cell(index, direction)?;
            if config.sand {
                let partition = f64::from(volumes.residual) / 2.0 / config.dt / config.dx;
                fields.qs[index] = (f64::from(fields.qs[index]) + partition) as f32;
                fields.qs[next] = (f64::from(fields.qs[next]) + partition) as f32;
            }
            index = next;
            let change = fields.change(index, volumes.residual, config);
            if change > 0.0 {
                fields.deposited[index] = (f64::from(fields.deposited[index]) + change) as f32;
            }
            volumes.residual = (f64::from(volumes.residual) - change) as f32;
            fields.update(index, change, config);
            let edge = fields.cells[index] == -1;
            if edge {
                volumes.exported += volumes.residual;
            }
            // Both tests run at an edge reached on stepmax, as in the reference.
            if step == config.stepmax {
                if config.force_deposit {
                    fields.deposited[index] += volumes.residual;
                    fields.update(index, f64::from(volumes.residual), config);
                    volumes.residual = 0.0;
                }
                volumes.lost += volumes.residual;
            }
            if edge {
                break;
            }
        }
    }
    Ok(volumes)
}

#[pyfunction]
pub(super) fn sediment_phase<'py>(
    py: Python<'py>,
    arrays: &Bound<'py, PyDict>,
    config: &Bound<'py, PyDict>,
    starts: PyReadonlyArray1<'py, i64>,
    seed: u64,
) -> PyResult<Bound<'py, PyDict>> {
    let config: Config = config.extract()?;
    config.validate()?;
    let mut fields = Fields::read(arrays, &config)?;
    let starts = starts.as_slice()?;
    if starts
        .iter()
        .any(|&s| s < 0 || s as usize >= fields.eta.len())
    {
        return Err(PyValueError::new_err("invalid sediment start index"));
    }
    let volumes = route(&mut fields, starts, &config, seed).map_err(PyValueError::new_err)?;
    let result = PyDict::new(py);
    let shape = (fields.rows, fields.cols);
    for (name, values) in [
        ("eta", fields.eta),
        ("depth", fields.depth),
        ("uw", fields.uw),
        ("ux", fields.ux),
        ("uy", fields.uy),
        (
            if config.sand {
                "Vp_dep_sand"
            } else {
                "Vp_dep_mud"
            },
            fields.deposited,
        ),
    ] {
        let array = Array2::from_shape_vec(shape, values)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        result.set_item(name, array.into_pyarray(py))?;
    }
    if config.sand {
        let array = Array2::from_shape_vec(shape, fields.qs)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        result.set_item("qs", array.into_pyarray(py))?;
    }
    let padded = Array2::from_shape_vec((fields.rows + 2, fields.cols + 2), fields.padded_depth)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    result.set_item("pad_depth", padded.into_pyarray(py))?;
    result.set_item("Vp_res", volumes.residual)?;
    result.set_item("Vp_exported", volumes.exported)?;
    result.set_item("Vp_lost", volumes.lost)?;
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn change_limits_keep_dry_deposition_zero_and_allow_erosion() {
        assert_eq!(limit_change(100.0, -2.0, 2.0, false, 1.0), 0.0);
        assert_eq!(limit_change(100.0, -2.0, 2.0, true, 1.0), 2.0);
        assert_eq!(limit_change(100.0, 2.0, 2.0, false, 0.5), 1.0);
    }
    #[test]
    fn padding_replicates_edges() {
        assert_eq!(
            pad_depth(&[1.0, 2.0], 1, 2),
            vec![1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0]
        );
    }
}
