*> pyDeltaRCM COBOL kernel - batched weighted parcel pick (ported routine).
*>
*> Port of shared_tools.random_pick over a batch of weight rows: float32
*> (COMP-1) cumulative weights, then a searchsorted(left) scan for the
*> first cutoff >= the row's uniform. The runtime calls random_pick once
*> per parcel per timestep (water_tools.py routing loop, sed_tools.py
*> parcel steps); this batched form is the parity/benchmark surface.
*>
*> Float discipline: every float op goes through one C IEEE helper --
*> cobol_faddf for the cumsum (COMP-1 COMPUTE is never used for float
*> arithmetic: GnuCOBOL lowers it to the GMP decimal runtime, which is
*> not bit-compatible with the numpy/numba float32 accumulation) and
*> cobol_fcmp_widen for comparisons (widens the float32 cutoff to double
*> and compares against the raw uniform double, exactly like numpy's
*> promotion inside searchsorted). Integer COMPUTE on COMP-5 /
*> BINARY-LONG is exact and used for indexing.
*>
*> Uniforms arrive pre-drawn by the Python adapter from the checkpointed
*> MT19937 stream as uniform(0, row_total) -- one draw per row, the same
*> values the jitted reference consumes (port contract C5: no RNG in
*> COBOL). WS-CUTOFFS is sized for 64 weights per row (the D8 window is
*> 9); the adapter rejects larger rows.
*>
*> Parity domain: weight rows are non-negative with a positive total
*> (model weight fields are sums of squared normalized gradients), so
*> the float32 cumsum is non-decreasing. The linear scan below is then
*> bit-equivalent to numpy searchsorted's binary search. Negative
*> weights would make the cumsum non-monotone and are outside the
*> model's and this port's parity guarantees.
       IDENTIFICATION DIVISION.
       PROGRAM-ID. RANDOM-PICK.
       DATA DIVISION.
       WORKING-STORAGE SECTION.
       01 WS-P PIC S9(9) COMP-5.
       01 WS-J PIC S9(9) COMP-5.
       01 WS-IDX PIC S9(9) COMP-5.
       01 WS-FOUND PIC 9.
       01 WS-ZERO COMP-2 VALUE 0.0.
       01 WS-CUM COMP-1.
       01 WS-V COMP-1.
       01 WS-CUTOFFS COMP-1 OCCURS 64 TIMES.
       01 WS-CMP PIC S9(9) COMP-5.
       LINKAGE SECTION.
       01 LS-K USAGE BINARY-LONG.
       01 LS-N USAGE BINARY-LONG.
       01 LS-ROWBUF.
          05 LS-W USAGE COMP-1 OCCURS 1 TO 268435455
                         DEPENDING ON LS-N.
       01 LS-RNBUF.
          05 LS-RN USAGE COMP-2 OCCURS 1 TO 268435455
                        DEPENDING ON LS-K.
       01 LS-OBUF.
          05 LS-OIDX USAGE BINARY-LONG OCCURS 1 TO 268435455
                         DEPENDING ON LS-K.
       PROCEDURE DIVISION USING LS-K LS-N LS-ROWBUF LS-RNBUF LS-OBUF.
       MAIN.
           PERFORM VARYING WS-P FROM 1 BY 1 UNTIL WS-P > LS-K
               MOVE 0 TO WS-CUM
               PERFORM VARYING WS-J FROM 1 BY 1 UNTIL WS-J > LS-N
                   COMPUTE WS-IDX = (WS-P - 1) * LS-N + WS-J
                   MOVE LS-W(WS-IDX) TO WS-V
                   IF WS-J = 1
                       *> np.cumsum writes out[0] = in[0] (no add);
                       *> 0 + v would differ for v = -0.0
                       MOVE WS-V TO WS-CUM
                   ELSE
                       CALL "cobol_faddf" USING WS-CUM WS-V WS-CUM
                   END-IF
                   MOVE WS-CUM TO WS-CUTOFFS(WS-J)
               END-PERFORM
               *> Error parity with the jitted reference: it raises
               *> ValueError unless the row total is > 0 (NaN included,
               *> since nan > 0 is false). cobol_fcmp_widen returns
               *> -1/0/+1 and 2 = unordered, so "not 1" covers zero,
               *> negative, and NaN totals.
               CALL "cobol_fcmp_widen" USING WS-CUTOFFS(LS-N) WS-ZERO
                                             WS-CMP
               IF WS-CMP NOT = 1
                   COMPUTE RETURN-CODE = 0 - WS-P
                   GOBACK
               END-IF
               *> searchsorted-left: first cutoff >= the uniform. Found
               *> means the compare is 0 (equal) or 1 (greater); 2
               *> (NaN uniform) must NOT match -- numpy searchsorted
               *> with a NaN value falls through to the sentinel. A
               *> uniform above every cutoff (float artifact at the
               *> draw limit) also falls through; the adapter maps the
               *> n sentinel to the reference's IndexError.
               COMPUTE LS-OIDX(WS-P) = LS-N
               MOVE 0 TO WS-FOUND
               PERFORM VARYING WS-J FROM 1 BY 1
                       UNTIL WS-J > LS-N OR WS-FOUND = 1
                   CALL "cobol_fcmp_widen" USING WS-CUTOFFS(WS-J)
                                                LS-RN(WS-P) WS-CMP
                   IF WS-CMP = 0 OR WS-CMP = 1
                       COMPUTE LS-OIDX(WS-P) = WS-J - 1
                       MOVE 1 TO WS-FOUND
                   END-IF
               END-PERFORM
           END-PERFORM
      *> The helper CALLs land their int returns in RETURN-CODE; zero
      *> it explicitly so the ctypes shim sees 0 on success.
           MOVE 0 TO RETURN-CODE
           GOBACK.
       END PROGRAM RANDOM-PICK.
