//! Rate the rooms a unit serves, each from all its diffusers.
//!
//! A receiver sits at ear height under each diffuser in a room. Every
//! diffuser in the room reaches it by Schultz's equation at its own distance,
//! and the powers add; the room is rated at its worst receiver. A terminal no
//! room holds is rated alone, in a stand-in room.

use duct_heal::ifc::IfcSpace;
use duct_heal::network::Point;
use wall_duct_core::autodiff::Dual;
use wall_duct_core::metrics::nc::nc_exact;
use wall_duct_core::metrics::octave::N_BANDS;
use wall_duct_core::physics::room::receiver_spl;

/// The stand-in room for a terminal no space holds.
pub const DEFAULT_ROOM_M3: f64 = 50.0;
pub const DEFAULT_RECEIVER_M: f64 = 2.0;
/// A seated listener's ear above the floor.
pub const EAR_HEIGHT_M: f64 = 1.2;

/// A terminal's sound power into its room and where it is (IFC coordinates).
pub struct Terminal {
    pub pwl: [f64; N_BANDS],
    pub location: Option<Point>,
    pub space: Option<usize>,
}

pub struct RoomRating {
    /// The space, or `None` for a terminal rated alone.
    pub space: Option<usize>,
    /// Indices into the terminals, in order.
    pub terminals: Vec<usize>,
    pub volume_m3: f64,
    /// The terminal the worst receiver is under, and the one that contributes
    /// most there, with its distance.
    pub receiver_under: usize,
    pub dominant: usize,
    pub dominant_distance_m: f64,
    pub spl: [f64; N_BANDS],
    pub nc: i32,
}

fn dist(a: Point, b: Point) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

fn at_receiver(pwl: &[f64; N_BANDS], volume_m3: f64, distance_m: f64) -> [f64; N_BANDS] {
    let c = Dual::<1>::constant;
    receiver_spl(&pwl.map(c), c(volume_m3), c(distance_m)).map(|b| b.value)
}

fn energy(spl: &[f64; N_BANDS]) -> f64 {
    spl.iter().map(|l| 10f64.powf(l / 10.0)).sum()
}

/// Rate every room `terminals` are in, and every terminal no room holds,
/// worst first (by NC, then loudest band).
pub fn rate(terminals: &[Terminal], spaces: &[IfcSpace]) -> Vec<RoomRating> {
    let mut rooms: Vec<(Option<usize>, Vec<usize>)> = Vec::new();
    for (i, t) in terminals.iter().enumerate() {
        match (t.space, t.location) {
            (Some(s), Some(_)) => match rooms.iter_mut().find(|(r, _)| *r == Some(s)) {
                Some((_, members)) => members.push(i),
                None => rooms.push((Some(s), vec![i])),
            },
            _ => rooms.push((None, vec![i])),
        }
    }
    let mut out: Vec<RoomRating> = rooms
        .into_iter()
        .map(|(space, members)| match space {
            None => {
                let i = members[0];
                let spl = at_receiver(&terminals[i].pwl, DEFAULT_ROOM_M3, DEFAULT_RECEIVER_M);
                RoomRating {
                    space,
                    terminals: members,
                    volume_m3: DEFAULT_ROOM_M3,
                    receiver_under: i,
                    dominant: i,
                    dominant_distance_m: DEFAULT_RECEIVER_M,
                    spl,
                    nc: nc_exact(&spl),
                }
            }
            Some(s) => {
                let room = &spaces[s];
                let volume = room.volume();
                let worst = members
                    .iter()
                    .map(|&r| {
                        let d = terminals[r].location.unwrap();
                        let ear = [d[0], d[1], room.min[2] + EAR_HEIGHT_M];
                        let parts: Vec<(usize, f64, [f64; N_BANDS])> = members
                            .iter()
                            .map(|&j| {
                                let at = dist(terminals[j].location.unwrap(), ear);
                                (j, at, at_receiver(&terminals[j].pwl, volume, at))
                            })
                            .collect();
                        let spl: [f64; N_BANDS] = std::array::from_fn(|b| {
                            10.0 * parts.iter().map(|p| 10f64.powf(p.2[b] / 10.0)).sum::<f64>().log10()
                        });
                        let (dominant, at, _) = *parts.iter().max_by(|a, b| energy(&a.2).total_cmp(&energy(&b.2))).unwrap();
                        (r, dominant, at, spl)
                    })
                    .max_by(|a, b| (nc_exact(&a.3), energy(&a.3)).partial_cmp(&(nc_exact(&b.3), energy(&b.3))).unwrap())
                    .unwrap();
                let (receiver_under, dominant, dominant_distance_m, spl) = worst;
                RoomRating { space, terminals: members, volume_m3: volume, receiver_under, dominant, dominant_distance_m, spl, nc: nc_exact(&spl) }
            }
        })
        .collect();
    let loudest = |r: &RoomRating| r.spl.iter().copied().fold(f64::MIN, f64::max);
    out.sort_by(|a, b| (b.nc, loudest(b)).partial_cmp(&(a.nc, loudest(a))).unwrap());
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const PWL: [f64; N_BANDS] = [60.0, 58.0, 55.0, 52.0, 50.0, 47.0, 44.0, 40.0];

    fn room() -> IfcSpace {
        IfcSpace { name: "Office".into(), min: [0.0, 0.0, 0.0], max: [10.0, 6.0, 2.7] }
    }

    /// Two diffusers at one point rate 3 dB over one: TM's rule for equal
    /// outlets equally far, 10 log10 n.
    #[test]
    fn equal_diffusers_add() {
        let at = |n: usize| {
            let t: Vec<Terminal> = (0..n).map(|_| Terminal { pwl: PWL, location: Some([5.0, 3.0, 2.7]), space: Some(0) }).collect();
            rate(&t, &[room()]).remove(0)
        };
        let (one, two) = (at(1), at(2));
        assert_eq!(two.terminals.len(), 2);
        for b in 0..N_BANDS {
            assert!((two.spl[b] - one.spl[b] - 10.0 * 2f64.log10()).abs() < 1e-9);
        }
        // Under the diffuser, 1.5 m from it.
        assert!((one.dominant_distance_m - 1.5).abs() < 1e-9);
        assert!((one.volume_m3 - 162.0).abs() < 1e-9);
    }

    /// The worst receiver is under the louder diffuser, and a far one adds
    /// less than an equal near one would.
    #[test]
    fn worst_receiver_and_distance() {
        let loud = PWL.map(|l| l + 6.0);
        let t = [
            Terminal { pwl: PWL, location: Some([1.0, 3.0, 2.7]), space: Some(0) },
            Terminal { pwl: loud, location: Some([9.0, 3.0, 2.7]), space: Some(0) },
        ];
        let r = rate(&t, &[room()]).remove(0);
        assert_eq!((r.receiver_under, r.dominant), (1, 1));
        let alone = rate(&t[1..], &[room()]).remove(0);
        let gain = r.spl[0] - alone.spl[0];
        assert!(gain > 0.0 && gain < 1.0, "the far quiet diffuser adds {gain} dB");
    }

    /// A terminal no room holds is rated alone in the stand-in room, and the
    /// worst room comes first.
    #[test]
    fn unplaced_terminal_alone() {
        let t = [
            Terminal { pwl: PWL, location: None, space: None },
            Terminal { pwl: PWL.map(|l| l + 20.0), location: None, space: None },
        ];
        let rated = rate(&t, &[]);
        assert_eq!(rated.len(), 2);
        assert_eq!((rated[0].terminals[0], rated[0].volume_m3, rated[0].dominant_distance_m), (1, DEFAULT_ROOM_M3, DEFAULT_RECEIVER_M));
        assert!(rated[0].nc >= rated[1].nc);
    }
}
