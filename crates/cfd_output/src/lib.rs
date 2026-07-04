//! VTK legacy (RECTILINEAR_GRID) output of the CFD state, driven entirely by the
//! `StructuredMesh` geometry — no `Grid` internals. Ported from toy-cfd's
//! `cfd_output`. Writes density, pressure, temperature, velocity, and Mach over
//! the interior cells.

use cfd_eos::{Eos, EosResource};
use cfd_state::CfdState;
use field_core::{FieldRegistry, StructuredMesh, UniformMesh};
use grass_app::prelude::*;
use std::fs::File;
use std::io::Write;

/// VTK-safe scientific notation: explicit exponent sign, ≥2 exponent digits, and
/// non-finite values written as zero (solid-derived primitives can be NaN).
fn fmt_e(x: f64) -> String {
    if !x.is_finite() {
        return "0.0000000000e+00".to_string();
    }
    let s = format!("{:.10e}", x);
    let pos = s.find('e').expect("scientific format contains 'e'");
    let (mantissa, exp_part) = s.split_at(pos);
    let exp_str = &exp_part[1..];
    let (sign, digits) = match exp_str.strip_prefix('-') {
        Some(rest) => ("-", rest),
        None => ("+", exp_str),
    };
    if digits.len() < 2 {
        format!("{mantissa}e{sign}0{digits}")
    } else {
        format!("{mantissa}e{sign}{digits}")
    }
}

/// Write the CFD state as VTK legacy ASCII to any sink. Iterates interior cells
/// in VTK's i-fastest (k-outer) order.
pub fn write_vtk_to<W: Write, M: StructuredMesh>(
    w: &mut W,
    mesh: &M,
    state: &CfdState,
    eos: &dyn Eos,
) -> std::io::Result<()> {
    let [ni, nj, nk] = mesh.dims();

    writeln!(w, "# vtk DataFile Version 3.0")?;
    writeln!(w, "dev_field_efvm output")?;
    writeln!(w, "ASCII")?;
    writeln!(w, "DATASET RECTILINEAR_GRID")?;
    writeln!(w, "DIMENSIONS {} {} {}", ni + 1, nj + 1, nk + 1)?;

    // Face coordinates per axis: face = center - half-width, plus the final face.
    let axis_faces = |axis: usize, n: usize, pick: &dyn Fn(usize) -> f64| -> Vec<f64> {
        let mut faces = Vec::with_capacity(n + 1);
        for m in 0..n {
            faces.push(pick(m) - 0.5 * mesh.spacing(axis, m));
        }
        faces.push(pick(n - 1) + 0.5 * mesh.spacing(axis, n - 1));
        faces
    };
    let cx = |i: usize| mesh.cell_centroid(mesh.idx(i, 0, 0))[0];
    let cy = |j: usize| mesh.cell_centroid(mesh.idx(0, j, 0))[1];
    let cz = |k: usize| mesh.cell_centroid(mesh.idx(0, 0, k))[2];

    for (label, n, faces) in [
        ("X", ni, axis_faces(0, ni, &cx)),
        ("Y", nj, axis_faces(1, nj, &cy)),
        ("Z", nk, axis_faces(2, nk, &cz)),
    ] {
        writeln!(w, "{label}_COORDINATES {} double", n + 1)?;
        for f in faces {
            writeln!(w, "{}", fmt_e(f))?;
        }
    }

    writeln!(w, "CELL_DATA {}", ni * nj * nk)?;

    // Helper to stream a scalar over cells in i-fastest order.
    macro_rules! scalar {
        ($name:literal, $cell:expr) => {{
            writeln!(w, "SCALARS {} double 1", $name)?;
            writeln!(w, "LOOKUP_TABLE default")?;
            for k in 0..nk {
                for j in 0..nj {
                    for i in 0..ni {
                        let u = &state.u[mesh.idx(i, j, k)];
                        writeln!(w, "{}", fmt_e($cell(u)))?;
                    }
                }
            }
        }};
    }

    scalar!("density", |u: &cfd_state::ConsVar| u.rho);
    scalar!("pressure", |u: &cfd_state::ConsVar| eos.pressure(u));
    scalar!("temperature", |u: &cfd_state::ConsVar| eos.cons_to_prim(u).t);

    writeln!(w, "VECTORS velocity double")?;
    for k in 0..nk {
        for j in 0..nj {
            for i in 0..ni {
                let v = state.u[mesh.idx(i, j, k)].velocity();
                writeln!(w, "{} {} {}", fmt_e(v[0]), fmt_e(v[1]), fmt_e(v[2]))?;
            }
        }
    }

    scalar!("mach", |u: &cfd_state::ConsVar| {
        let v = u.velocity();
        let vmag = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        vmag / eos.sound_speed(u)
    });

    Ok(())
}

/// Path-based wrapper around [`write_vtk_to`].
pub fn write_vtk<M: StructuredMesh>(
    path: &str,
    mesh: &M,
    state: &CfdState,
    eos: &dyn Eos,
) -> std::io::Result<()> {
    let mut f = File::create(path)?;
    write_vtk_to(&mut f, mesh, state, eos)
}

/// Writes a single VTK file at the end of the run (via a cleanup hook).
pub struct OutputPlugin {
    pub path: String,
}

impl Default for OutputPlugin {
    fn default() -> Self {
        Self { path: "output.vtk".to_string() }
    }
}

impl Plugin for OutputPlugin {
    fn build(&self, app: &mut App) {
        let path = self.path.clone();
        app.add_cleanup_with_app(move |app: &mut App| {
            let mesh = app.get_resource_ref::<UniformMesh>().expect("mesh");
            let reg = app.get_resource_ref::<FieldRegistry>().expect("registry");
            let eos = app.get_resource_ref::<EosResource>().expect("eos");
            let state = reg.get::<CfdState>().expect("CfdState");
            if let Err(e) = write_vtk(&path, &*mesh, &state, &*eos.0) {
                eprintln!("OutputPlugin: failed to write {path}: {e}");
            } else {
                println!("OutputPlugin: wrote {path}");
            }
        });
    }
    fn provides(&self) -> Vec<&str> {
        vec!["output"]
    }
    fn requires(&self) -> Vec<&str> {
        vec!["cfd_state", "eos", "mesh"]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cfd_eos::IdealGas;
    use cfd_state::PrimVar;
    use field_core::{FieldData, FvMesh, UniformMeshConfig};

    #[test]
    fn writes_well_formed_vtk() {
        let mesh = UniformMesh::from_config(&UniformMeshConfig {
            nx: 4,
            ny: 3,
            nz: 2,
            ng: 2,
            bounds_lo: [0.0; 3],
            bounds_hi: [1.0, 1.0, 1.0],
            y_edges: None,
            z_edges: None,
        });
        let eos = IdealGas::air();
        let mut state = CfdState::default();
        state.resize(mesh.n_cells_total());
        for u in state.u.iter_mut() {
            *u = eos.prim_to_cons(&PrimVar::new(1.0, 10.0, 0.0, 0.0, 1e5, 348.0));
        }

        let mut buf = Vec::new();
        write_vtk_to(&mut buf, &mesh, &state, &eos).unwrap();
        let s = String::from_utf8(buf).unwrap();

        assert!(s.contains("DATASET RECTILINEAR_GRID"));
        assert!(s.contains("DIMENSIONS 5 4 3")); // (ni+1, nj+1, nk+1)
        assert!(s.contains("CELL_DATA 24")); // 4*3*2
        // One density value per interior cell.
        let density_block = s.split("SCALARS density double 1").nth(1).unwrap();
        let n_density = density_block
            .lines()
            .take_while(|l| !l.starts_with("SCALARS") && !l.starts_with("VECTORS"))
            // numeric value lines start with a digit or minus sign.
            .filter(|l| l.starts_with(|c: char| c.is_ascii_digit() || c == '-'))
            .count();
        assert_eq!(n_density, 24);
    }
}
