//! Duct-domain elements from an IFC file, and their merge into a network
//! built from the Revit file, or into an empty one when there is no Revit file.
//!
//! An IFC export fills what rvt-rs leaves: geometry where it decodes none,
//! accessories it does not decode, and IFC's own class for each element.
//! Revit writes its element id as the IFC Tag, so the two join by id. Port
//! connections, where the export has them, are recorded joins. Revit's
//! default export has none, so an element no join reaches is joined to the
//! ducts that end inside it.
//!
//! Reads ISO 10303-21 text: entities are parsed on demand, geometry is only
//! as much as the network needs (an axis and section for straight ducts, the
//! extent of everything else in its own frame), in metres.

use std::collections::{HashMap, HashSet};

use crate::build::CFS;
use crate::network::{EdgeSource, IfcInfo, Network, Node, NodeKind, Point};

/// Duct-domain classes, IFC4 and their IFC2x3 equivalents' type names.
const DUCT_CLASSES: &[&str] = &[
    "IFCDUCTSEGMENT",
    "IFCDUCTFITTING",
    "IFCAIRTERMINAL",
    "IFCAIRTERMINALBOX",
    "IFCDAMPER",
    "IFCDUCTSILENCER",
    "IFCFAN",
    "IFCUNITARYEQUIPMENT",
    "IFCAIRTOAIRHEATRECOVERY",
    "IFCFILTER",
];

/// IFC2x3's generic occurrence classes: what they are is in their type.
const GENERIC_CLASSES: &[&str] = &[
    "IFCFLOWSEGMENT",
    "IFCFLOWFITTING",
    "IFCFLOWTERMINAL",
    "IFCFLOWCONTROLLER",
    "IFCFLOWMOVINGDEVICE",
    "IFCENERGYCONVERSIONDEVICE",
    "IFCFLOWTREATMENTDEVICE",
];

#[derive(Debug, Clone)]
pub struct IfcElement {
    /// The authoring tool's element id; Revit's, for a Revit export.
    pub tag: Option<i64>,
    /// Specific class, upper case ("IFCDUCTSILENCER").
    pub class: String,
    /// Predefined type, from the occurrence or else its type; "" if neither.
    pub predefined: String,
    pub name: String,
    pub description: String,
    pub shape: Option<Shape>,
    /// System names from the "System Name" property and system groups.
    pub systems: Vec<String>,
    /// Design airflow, m^3/s, from a "Flow" or "Air Flow" property, else the
    /// largest typed flow.
    pub flow: Option<f64>,
    /// Elements (indices into `parse`'s result) a port of this one connects to.
    pub joined: Vec<usize>,
}

#[derive(Debug, Clone, Copy)]
pub enum Shape {
    /// An extruded straight: centreline and section, metres.
    Axis { start: Point, end: Point, width: f64, height: f64 },
    /// Anything else: its frame and the extent of its geometry in it.
    Box { frame: Frame, min: Point, max: Point },
}

/// A placement: unit axes and origin, in world coordinates.
#[derive(Debug, Clone, Copy)]
pub struct Frame {
    pub x: Point,
    pub y: Point,
    pub z: Point,
    pub o: Point,
}

fn add(a: Point, b: Point) -> Point {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}
fn sub(a: Point, b: Point) -> Point {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn mul(a: Point, k: f64) -> Point {
    [a[0] * k, a[1] * k, a[2] * k]
}
fn dot(a: Point, b: Point) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn cross(a: Point, b: Point) -> Point {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}
fn unit(a: Point) -> Point {
    mul(a, 1.0 / dot(a, a).sqrt())
}

const X: Point = [1.0, 0.0, 0.0];
const Y: Point = [0.0, 1.0, 0.0];
const Z: Point = [0.0, 0.0, 1.0];

impl Frame {
    const WORLD: Frame = Frame { x: X, y: Y, z: Z, o: [0.0; 3] };

    fn rotate(&self, v: Point) -> Point {
        add(add(mul(self.x, v[0]), mul(self.y, v[1])), mul(self.z, v[2]))
    }

    pub fn apply(&self, p: Point) -> Point {
        add(self.o, self.rotate(p))
    }

    /// `inner`, given in this frame, in world coordinates.
    fn then(&self, inner: &Frame) -> Frame {
        Frame { x: self.rotate(inner.x), y: self.rotate(inner.y), z: self.rotate(inner.z), o: self.apply(inner.o) }
    }

    /// World point `p` in this frame's coordinates.
    pub fn local(&self, p: Point) -> Point {
        let d = sub(p, self.o);
        [dot(d, self.x), dot(d, self.y), dot(d, self.z)]
    }
}

#[derive(Debug)]
enum Arg<'a> {
    Ref(u64),
    Num(f64),
    Str(&'a str),
    Enum(&'a str),
    List(Vec<Arg<'a>>),
    Null,
}

impl Arg<'_> {
    fn id(&self) -> Option<u64> {
        match self {
            Arg::Ref(r) => Some(*r),
            _ => None,
        }
    }
    fn num(&self) -> Option<f64> {
        match self {
            Arg::Num(v) => Some(*v),
            _ => None,
        }
    }
    fn list(&self) -> &[Self] {
        match self {
            Arg::List(l) => l,
            _ => &[],
        }
    }
}

/// One entity's argument list, parsed.
struct Args<'a> {
    s: &'a str,
    i: usize,
}

impl<'a> Args<'a> {
    fn parse(s: &'a str) -> Vec<Arg<'a>> {
        Args { s, i: 0 }.items()
    }

    fn peek(&self) -> u8 {
        self.s.as_bytes().get(self.i).copied().unwrap_or(b')')
    }

    fn skip_ws(&mut self) {
        while self.peek().is_ascii_whitespace() {
            self.i += 1;
        }
    }

    fn take_while(&mut self, f: impl Fn(u8) -> bool) -> &'a str {
        let start = self.i;
        while self.i < self.s.len() && f(self.s.as_bytes()[self.i]) {
            self.i += 1;
        }
        &self.s[start..self.i]
    }

    /// Comma-separated items up to the closing bracket (or the end).
    fn items(&mut self) -> Vec<Arg<'a>> {
        let mut out = Vec::new();
        loop {
            self.skip_ws();
            if self.peek() == b')' {
                return out;
            }
            out.push(self.item());
            self.skip_ws();
            if self.peek() == b',' {
                self.i += 1;
            }
        }
    }

    fn bracketed(&mut self) -> Vec<Arg<'a>> {
        self.i += 1;
        let items = self.items();
        self.i += 1;
        items
    }

    fn item(&mut self) -> Arg<'a> {
        match self.peek() {
            b'#' => {
                self.i += 1;
                Arg::Ref(self.take_while(|c| c.is_ascii_digit()).parse().unwrap_or(0))
            }
            b'\'' => {
                self.i += 1;
                let start = self.i;
                let bytes = self.s.as_bytes();
                while self.i < bytes.len() {
                    if bytes[self.i] == b'\'' {
                        if bytes.get(self.i + 1) == Some(&b'\'') {
                            self.i += 2;
                            continue;
                        }
                        break;
                    }
                    self.i += 1;
                }
                let s = &self.s[start..self.i];
                self.i += 1;
                Arg::Str(s)
            }
            b'.' => {
                self.i += 1;
                let e = self.take_while(|c| c != b'.');
                self.i += 1;
                Arg::Enum(e)
            }
            b'(' => Arg::List(self.bracketed()),
            b'$' | b'*' => {
                self.i += 1;
                Arg::Null
            }
            c if c.is_ascii_alphabetic() => {
                // A typed value, IFCLABEL('x') or IFCARCINDEX((1,2,3)): keep its content.
                self.take_while(|c| c.is_ascii_alphanumeric() || c == b'_');
                self.skip_ws();
                if self.peek() != b'(' {
                    return Arg::Null;
                }
                let mut inner = self.bracketed();
                if inner.len() == 1 { inner.pop().unwrap() } else { Arg::List(inner) }
            }
            _ => Arg::Num(self.take_while(|c| c != b',' && c != b')').trim().parse().unwrap_or(f64::NAN)),
        }
    }
}

/// STEP string escapes: '' and the \X\hh and \X2\hhhh..\X0\ code points.
fn text(raw: &str) -> String {
    let raw = raw.replace("''", "'");
    let mut out = String::new();
    let mut rest = raw.as_str();
    while let Some(i) = rest.find('\\') {
        out.push_str(&rest[..i]);
        let r = &rest[i..];
        if let Some(c) = r.strip_prefix("\\X\\").and_then(|h| h.get(..2)).and_then(|h| u8::from_str_radix(h, 16).ok()) {
            out.push(c as char);
            rest = &r[5..];
        } else if let Some((hex, after)) = r.strip_prefix("\\X2\\").and_then(|h| h.split_once("\\X0\\")) {
            for quad in hex.as_bytes().chunks(4) {
                let c = std::str::from_utf8(quad).ok().and_then(|q| u32::from_str_radix(q, 16).ok()).and_then(char::from_u32);
                out.extend(c);
            }
            rest = after;
        } else {
            out.push('\\');
            rest = &r[1..];
        }
    }
    out.push_str(rest);
    out
}

struct File<'a> {
    entities: HashMap<u64, (&'a str, &'a str)>,
    /// Metres per file length unit.
    scale: f64,
    /// m^3/s per file flow unit.
    flow_scale: f64,
}

/// Points and extrusions of an element's body, in its placement's frame.
#[derive(Default)]
struct Body {
    points: Vec<Point>,
    extrusions: Vec<(Point, Point, f64, f64)>,
}

impl<'a> File<'a> {
    fn new(src: &'a str) -> Self {
        let mut entities = HashMap::new();
        let data = src.find("DATA;").map_or(src, |i| &src[i + 5..]);
        let bytes = data.as_bytes();
        let mut i = 0;
        while let Some(at) = data[i..].find('#').map(|k| k + i) {
            // The statement runs to the first ';' outside a string.
            let mut end = at;
            let mut quoted = false;
            while end < bytes.len() && (quoted || bytes[end] != b';') {
                if bytes[end] == b'\'' {
                    quoted = !quoted;
                }
                end += 1;
            }
            let stmt = &data[at + 1..end];
            i = end + 1;
            let Some((id, rest)) = stmt.split_once('=') else { continue };
            let Some((class, args)) = rest.trim_start().split_once('(') else { continue };
            let Ok(id) = id.trim().parse() else { continue };
            entities.insert(id, (class.trim(), args.trim_end().strip_suffix(')').unwrap_or(args)));
        }
        let mut file = File { entities, scale: 1.0, flow_scale: 1.0 };
        file.scale = file.length_unit();
        file.flow_scale = file.project_unit("VOLUMETRICFLOWRATEUNIT").and_then(|u| file.unit_factor(u)).unwrap_or(1.0);
        file
    }

    fn project_unit(&self, kind: &str) -> Option<u64> {
        let (_, assignment) = self.of_class("IFCUNITASSIGNMENT").next()?;
        assignment.first()?.list().iter().filter_map(Arg::id).find(|&u| {
            self.get(u).is_some_and(|(_, a)| a.iter().any(|x| matches!(x, Arg::Enum(k) if *k == kind)))
        })
    }

    /// SI base units per one of the unit `id`: prefixed SI, derived and
    /// conversion-based units (L/s, m^3/h, ft^3/min).
    fn unit_factor(&self, id: u64) -> Option<f64> {
        let (class, a) = self.get(id)?;
        match class {
            "IFCSIUNIT" => {
                let prefix = match a.get(2) {
                    Some(Arg::Enum("KILO")) => 1e3,
                    Some(Arg::Enum("DECI")) => 1e-1,
                    Some(Arg::Enum("CENTI")) => 1e-2,
                    Some(Arg::Enum("MILLI")) => 1e-3,
                    _ => 1.0,
                };
                let power = match a.get(1) {
                    Some(Arg::Enum("VOLUMEUNIT")) => 3,
                    Some(Arg::Enum("AREAUNIT")) => 2,
                    _ => 1,
                };
                Some(f64::powi(prefix, power))
            }
            "IFCDERIVEDUNIT" => a.first()?.list().iter().try_fold(1.0, |k, e| {
                let (_, e) = self.get(e.id()?)?;
                Some(k * self.unit_factor(e.first()?.id()?)?.powf(e.get(1)?.num()?))
            }),
            "IFCCONVERSIONBASEDUNIT" => {
                let (_, m) = self.get(a.get(3)?.id()?)?;
                Some(m.first()?.num()? * self.unit_factor(m.get(1)?.id()?)?)
            }
            _ => None,
        }
    }

    fn get(&self, id: u64) -> Option<(&'a str, Vec<Arg<'a>>)> {
        self.entities.get(&id).map(|(class, args)| (*class, Args::parse(args)))
    }

    fn of_class(&self, class: &str) -> impl Iterator<Item = (u64, Vec<Arg<'a>>)> + '_ {
        let class = class.to_owned();
        self.entities.iter().filter(move |(_, (c, _))| *c == class).map(|(id, (_, args))| (*id, Args::parse(args)))
    }

    /// Metres per length unit, from the project's unit assignment.
    fn length_unit(&self) -> f64 {
        for (_, assignment) in self.of_class("IFCUNITASSIGNMENT") {
            for unit in assignment.first().map(Arg::list).unwrap_or(&[]) {
                let Some((class, a)) = unit.id().and_then(|id| self.get(id)) else { continue };
                if !matches!(a.get(1), Some(Arg::Enum("LENGTHUNIT"))) {
                    continue;
                }
                return match (class, a.get(2)) {
                    ("IFCSIUNIT", Some(Arg::Enum("MILLI"))) => 0.001,
                    ("IFCSIUNIT", Some(Arg::Enum("CENTI"))) => 0.01,
                    ("IFCCONVERSIONBASEDUNIT", Some(Arg::Str(name))) if name.eq_ignore_ascii_case("inch") => 0.0254,
                    ("IFCCONVERSIONBASEDUNIT", Some(Arg::Str(name))) if name.eq_ignore_ascii_case("foot") => 0.3048,
                    _ => 1.0,
                };
            }
        }
        1.0
    }

    fn coords(&self, arg: Option<&Arg>) -> Option<Vec<f64>> {
        let (_, a) = self.get(arg?.id()?)?;
        Some(a.first()?.list().iter().filter_map(Arg::num).collect())
    }

    fn point(&self, arg: Option<&Arg>) -> Option<Point> {
        let c = self.coords(arg)?;
        Some(mul([*c.first()?, *c.get(1)?, c.get(2).copied().unwrap_or(0.0)], self.scale))
    }

    fn direction(&self, arg: Option<&Arg>) -> Option<Point> {
        let c = self.coords(arg)?;
        Some(unit([*c.first()?, *c.get(1)?, c.get(2).copied().unwrap_or(0.0)]))
    }

    fn axis_placement(&self, id: Option<u64>) -> Frame {
        let Some((class, a)) = id.and_then(|id| self.get(id)) else { return Frame::WORLD };
        let o = self.point(a.first()).unwrap_or([0.0; 3]);
        let (z, r) = match class {
            "IFCAXIS2PLACEMENT3D" => (self.direction(a.get(1)).unwrap_or(Z), self.direction(a.get(2))),
            _ => (Z, self.direction(a.get(1))),
        };
        let x = unit(sub(r.unwrap_or(X), mul(z, dot(r.unwrap_or(X), z))));
        Frame { x, y: cross(z, x), z, o }
    }

    fn placement(&self, id: Option<u64>) -> Frame {
        match id.and_then(|id| self.get(id)) {
            Some(("IFCLOCALPLACEMENT", a)) => {
                let parent = self.placement(a.first().and_then(Arg::id));
                parent.then(&self.axis_placement(a.get(1).and_then(Arg::id)))
            }
            _ => Frame::WORLD,
        }
    }

    /// The 2D extent of a profile, file units scaled.
    fn profile_extent(&self, id: Option<u64>) -> Option<([f64; 2], [f64; 2])> {
        let (class, a) = self.get(id?)?;
        // Parametric profiles are centred on their position (argument 2).
        let centre = || self.axis_placement(a.get(2).and_then(Arg::id)).o;
        let half = match class {
            "IFCRECTANGLEPROFILEDEF" | "IFCRECTANGLEHOLLOWPROFILEDEF" => {
                [a.get(3)?.num()? * self.scale / 2.0, a.get(4)?.num()? * self.scale / 2.0]
            }
            "IFCCIRCLEPROFILEDEF" | "IFCCIRCLEHOLLOWPROFILEDEF" => {
                let r = a.get(3)?.num()? * self.scale;
                [r, r]
            }
            "IFCARBITRARYCLOSEDPROFILEDEF" | "IFCARBITRARYPROFILEDEFWITHVOIDS" => {
                let mut points = Vec::new();
                self.collect_points(a.get(2)?.id()?, &Frame::WORLD, &mut points, &mut HashSet::new());
                let (min, max) = extent(&points)?;
                return Some(([min[0], min[1]], [max[0], max[1]]));
            }
            _ => return None,
        };
        let c = centre();
        Some(([c[0] - half[0], c[1] - half[1]], [c[0] + half[0], c[1] + half[1]]))
    }

    /// Every point reachable from `id` (face sets, breps, polylines), in `frame`.
    fn collect_points(&self, id: u64, frame: &Frame, out: &mut Vec<Point>, seen: &mut HashSet<u64>) {
        if !seen.insert(id) {
            return;
        }
        let Some((class, a)) = self.get(id) else { return };
        match class {
            "IFCCARTESIANPOINT" => out.extend(self.point(Some(&Arg::Ref(id))).map(|p| frame.apply(p))),
            "IFCCARTESIANPOINTLIST3D" | "IFCCARTESIANPOINTLIST2D" => {
                for c in a.first().map(Arg::list).unwrap_or(&[]) {
                    let c: Vec<f64> = c.list().iter().filter_map(Arg::num).collect();
                    if c.len() >= 2 {
                        out.push(frame.apply(mul([c[0], c[1], c.get(2).copied().unwrap_or(0.0)], self.scale)));
                    }
                }
            }
            _ => {
                fn refs(a: &Arg, out: &mut Vec<u64>) {
                    match a {
                        Arg::Ref(r) => out.push(*r),
                        Arg::List(l) => l.iter().for_each(|x| refs(x, out)),
                        _ => {}
                    }
                }
                let mut next = Vec::new();
                a.iter().for_each(|x| refs(x, &mut next));
                for r in next {
                    self.collect_points(r, frame, out, seen);
                }
            }
        }
    }

    /// A representation item's geometry, in the element's frame.
    fn item(&self, id: u64, frame: &Frame, body: &mut Body) {
        let Some((class, a)) = self.get(id) else { return };
        match class {
            "IFCMAPPEDITEM" => {
                let Some((_, map)) = a.first().and_then(Arg::id).and_then(|m| self.get(m)) else { return };
                let target = match a.get(1).and_then(Arg::id).and_then(|t| self.get(t)) {
                    Some((_, t)) => Frame {
                        x: self.direction(t.first()).unwrap_or(X),
                        y: self.direction(t.get(1)).unwrap_or(Y),
                        z: self.direction(t.get(4)).unwrap_or(Z),
                        o: self.point(t.get(2)).unwrap_or([0.0; 3]),
                    },
                    None => Frame::WORLD,
                };
                let inner = frame.then(&target).then(&self.axis_placement(map.first().and_then(Arg::id)));
                if let Some((_, rep)) = map.get(1).and_then(Arg::id).and_then(|r| self.get(r)) {
                    for item in rep.get(3).map(Arg::list).unwrap_or(&[]).iter().filter_map(Arg::id) {
                        self.item(item, &inner, body);
                    }
                }
            }
            "IFCEXTRUDEDAREASOLID" => {
                let Some((min, max)) = self.profile_extent(a.first().and_then(Arg::id)) else { return };
                let at = frame.then(&self.axis_placement(a.get(1).and_then(Arg::id)));
                let (Some(dir), Some(depth)) = (self.direction(a.get(2)), a.get(3).and_then(Arg::num)) else { return };
                let run = mul(at.rotate(dir), depth * self.scale);
                let start = at.apply([(min[0] + max[0]) / 2.0, (min[1] + max[1]) / 2.0, 0.0]);
                body.extrusions.push((start, add(start, run), max[0] - min[0], max[1] - min[1]));
                for corner in [[min[0], min[1]], [min[0], max[1]], [max[0], min[1]], [max[0], max[1]]] {
                    let p = at.apply([corner[0], corner[1], 0.0]);
                    body.points.extend([p, add(p, run)]);
                }
            }
            "IFCBOOLEANRESULT" | "IFCBOOLEANCLIPPINGRESULT" => {
                if let Some(first) = a.get(1).and_then(Arg::id) {
                    self.item(first, frame, body);
                }
            }
            _ => self.collect_points(id, frame, &mut body.points, &mut HashSet::new()),
        }
    }

    /// The element's body geometry, in its placement's frame.
    fn body(&self, shape: Option<u64>) -> Body {
        let mut body = Body::default();
        let Some((_, a)) = shape.and_then(|s| self.get(s)) else { return body };
        let reps: Vec<_> = a.get(2).map(Arg::list).unwrap_or(&[]).iter().filter_map(|r| self.get(r.id()?)).collect();
        let is_body = |rep: &Vec<Arg>| matches!(rep.get(1), Some(Arg::Str("Body")));
        let any_body = reps.iter().any(|(_, r)| is_body(r));
        for (_, rep) in reps.iter().filter(|(_, r)| !any_body || is_body(r)) {
            for item in rep.get(3).map(Arg::list).unwrap_or(&[]).iter().filter_map(Arg::id) {
                self.item(item, &Frame::WORLD, &mut body);
            }
        }
        body
    }

    /// A single-value property the network uses, onto `e`. Revit writes its
    /// parameters as bare reals in its internal units, so an IFCREAL flow is
    /// ft^3/s; a typed flow is in the project's flow unit. Returns any other
    /// typed flow, m^3/s: other tools name theirs in the model's language.
    fn property(&self, id: u64, e: &mut IfcElement) -> Option<f64> {
        let Some(("IFCPROPERTYSINGLEVALUE", raw)) = self.entities.get(&id).copied() else { return None };
        let a = Args::parse(raw);
        let typed = raw.contains("IFCVOLUMETRICFLOWRATEMEASURE(");
        let flow = |v: f64| {
            if raw.contains("IFCREAL(") {
                Some(v * CFS)
            } else if typed {
                Some(v * self.flow_scale)
            } else {
                None
            }
        };
        match (a.first(), a.get(2)) {
            (Some(Arg::Str("System Name")), Some(Arg::Str(v))) => {
                e.systems.extend(text(v).split(',').map(str::trim).filter(|s| !s.is_empty()).map(String::from))
            }
            (Some(Arg::Str("Flow")), Some(Arg::Num(v))) => e.flow = flow(*v).or(e.flow),
            (Some(Arg::Str("Air Flow")), Some(Arg::Num(v))) if e.flow.is_none() => e.flow = flow(*v),
            (_, Some(Arg::Num(v))) if typed => return flow(*v),
            _ => {}
        }
        None
    }
}

fn extent(points: &[Point]) -> Option<(Point, Point)> {
    let first = *points.first()?;
    Some(points.iter().fold((first, first), |(lo, hi), p| {
        ([lo[0].min(p[0]), lo[1].min(p[1]), lo[2].min(p[2])], [hi[0].max(p[0]), hi[1].max(p[1]), hi[2].max(p[2])])
    }))
}

/// The duct-domain elements of an IFC file.
pub fn parse(bytes: &[u8]) -> Vec<IfcElement> {
    let src = String::from_utf8_lossy(bytes);
    let file = File::new(&src);

    let mut type_of = HashMap::new();
    for (_, a) in file.of_class("IFCRELDEFINESBYTYPE") {
        let Some(ty) = a.get(5).and_then(Arg::id) else { continue };
        for occurrence in a.get(4).map(Arg::list).unwrap_or(&[]).iter().filter_map(Arg::id) {
            type_of.insert(occurrence, ty);
        }
    }
    let predefined = |a: &[Arg]| match a.last() {
        Some(Arg::Enum(e)) if !matches!(*e, "NOTDEFINED" | "USERDEFINED") => e.to_string(),
        _ => String::new(),
    };

    let mut ids: Vec<u64> = file
        .entities
        .iter()
        .filter(|(_, (c, _))| DUCT_CLASSES.contains(c) || GENERIC_CLASSES.contains(c))
        .map(|(id, _)| *id)
        .collect();
    ids.sort_unstable();
    let mut out = Vec::new();
    let mut index = HashMap::new();
    for id in ids {
        let (class, a) = file.get(id).expect("listed id");
        let ty = type_of.get(&id).and_then(|t| file.get(*t));
        let class = match (GENERIC_CLASSES.contains(&class), &ty) {
            (false, _) => class.to_string(),
            (true, Some((ty_class, _))) => match ty_class.strip_suffix("TYPE") {
                Some(c) if DUCT_CLASSES.contains(&c) => c.to_string(),
                _ => continue,
            },
            (true, None) => continue,
        };
        let mut pre = predefined(&a);
        if pre.is_empty() {
            pre = ty.as_ref().map(|(_, t)| predefined(t)).unwrap_or_default();
        }
        let s = |k: usize| match a.get(k) {
            Some(Arg::Str(v)) => text(v),
            _ => String::new(),
        };
        let place = file.placement(a.get(5).and_then(Arg::id));
        let body = file.body(a.get(6).and_then(Arg::id));
        let shape = match body.extrusions.first() {
            Some(&(start, end, width, height)) if class == "IFCDUCTSEGMENT" && body.extrusions.len() == 1 => {
                Some(Shape::Axis { start: place.apply(start), end: place.apply(end), width, height })
            }
            _ => extent(&body.points).map(|(min, max)| Shape::Box { frame: place, min, max }),
        };
        index.insert(id, out.len());
        out.push(IfcElement {
            tag: s(7).trim().parse().ok(),
            class,
            predefined: pre,
            name: s(2),
            description: s(3),
            shape,
            systems: Vec::new(),
            flow: None,
            joined: Vec::new(),
        });
    }

    let mut other_flow = HashMap::<usize, f64>::new();
    for (_, a) in file.of_class("IFCRELDEFINESBYPROPERTIES") {
        let Some((_, set)) = a.get(5).and_then(Arg::id).and_then(|s| file.get(s)) else { continue };
        for &k in a.get(4).map(Arg::list).unwrap_or(&[]).iter().filter_map(|o| index.get(&o.id()?)) {
            for p in set.get(4).map(Arg::list).unwrap_or(&[]).iter().filter_map(Arg::id) {
                if let Some(q) = file.property(p, &mut out[k]) {
                    let most = other_flow.entry(k).or_default();
                    *most = most.max(q);
                }
            }
        }
    }
    // Without a named flow, the largest typed one: a terminal's design flow
    // beside its measured or minimum ones.
    for (k, q) in other_flow {
        if out[k].flow.is_none() && q > 0.0 {
            out[k].flow = Some(q);
        }
    }
    // Membership of a system group, IFC's own record of what a "System Name" says.
    for (_, a) in file.of_class("IFCRELASSIGNSTOGROUP") {
        let Some(("IFCSYSTEM" | "IFCDISTRIBUTIONSYSTEM", g)) = a.get(6).and_then(Arg::id).and_then(|g| file.get(g)) else { continue };
        let Some(Arg::Str(name)) = g.get(2) else { continue };
        let name = text(name);
        for &k in a.get(4).map(Arg::list).unwrap_or(&[]).iter().filter_map(|o| index.get(&o.id()?)) {
            if !out[k].systems.contains(&name) {
                out[k].systems.push(name.clone());
            }
        }
    }

    // A port's element: IFC2x3 relates them, IFC4 nests ports in the element.
    let mut owner = HashMap::new();
    for (_, a) in file.of_class("IFCRELCONNECTSPORTTOELEMENT") {
        if let (Some(port), Some(element)) = (a.get(4).and_then(Arg::id), a.get(5).and_then(Arg::id)) {
            owner.insert(port, element);
        }
    }
    for (_, a) in file.of_class("IFCRELNESTS") {
        let Some(element) = a.get(4).and_then(Arg::id) else { continue };
        for port in a.get(5).map(Arg::list).unwrap_or(&[]).iter().filter_map(Arg::id) {
            if file.entities.get(&port).is_some_and(|(c, _)| *c == "IFCDISTRIBUTIONPORT") {
                owner.insert(port, element);
            }
        }
    }
    let element_of = |a: Option<&Arg>| a.and_then(Arg::id).and_then(|p| owner.get(&p)).and_then(|e| index.get(e)).copied();
    for (_, a) in file.of_class("IFCRELCONNECTSPORTS") {
        let (Some(x), Some(y)) = (element_of(a.get(4)), element_of(a.get(5))) else { continue };
        // A realizing element (a flex duct, say) sits between the two.
        match a.get(6).and_then(Arg::id).and_then(|r| index.get(&r)).copied() {
            Some(r) => {
                out[x].joined.push(r);
                out[r].joined.push(y);
            }
            None => out[x].joined.push(y),
        }
    }
    out
}

fn kind_of_class(class: &str) -> NodeKind {
    match class {
        "IFCDUCTSEGMENT" => NodeKind::DuctNoGeometry,
        "IFCAIRTERMINAL" => NodeKind::Terminal,
        "IFCFAN" | "IFCUNITARYEQUIPMENT" | "IFCAIRTOAIRHEATRECOVERY" => NodeKind::Equipment,
        _ => NodeKind::Fitting,
    }
}

#[derive(Debug, Default)]
pub struct MergeReport {
    pub elements: usize,
    /// IFC elements whose tag is a node's id.
    pub matched: usize,
    /// Ducts given a centreline and section by the IFC.
    pub geometry: usize,
    /// Elements only the IFC has, added as nodes.
    pub added: usize,
    /// Joins the IFC's port connections record.
    pub ports: usize,
    /// Duct ends joined to an element they lie inside.
    pub joined: usize,
}

/// A duct end this close outside an element's extent still counts as inside.
const INSIDE_TOL: f64 = 0.02;

/// Merge IFC elements into `net`: geometry and IFC class onto the nodes they
/// tag, new nodes for the rest, and the joins their ports record. A node no
/// join reaches is then joined to every duct that ends inside it. Merged into
/// an empty network, this is the network of an IFC on its own.
pub fn merge(net: &mut Network, elements: &[IfcElement]) -> MergeReport {
    let mut report = MergeReport { elements: elements.len(), ..Default::default() };
    let mut boxes = Vec::new();
    let mut nodes = Vec::with_capacity(elements.len());
    for (k, e) in elements.iter().enumerate() {
        let i = match e.tag.and_then(|t| net.find(t)) {
            Some(i) => {
                report.matched += 1;
                i
            }
            None => {
                report.added += 1;
                net.add_node(Node {
                    id: e.tag.unwrap_or(-(k as i64) - 1),
                    kind: NodeKind::Undecoded,
                    family: String::new(),
                    type_name: String::new(),
                    systems: Vec::new(),
                    location: None,
                    flow: None,
                    ifc: None,
                })
            }
        };
        nodes.push(i);
        let node = &mut net.nodes[i];
        if node.kind == NodeKind::Undecoded {
            node.kind = kind_of_class(&e.class);
            // Revit names an occurrence "Family:Type:Type:id".
            let mut name = e.name.split(':');
            node.family = name.next().unwrap_or_default().to_string();
            node.type_name = name.next().unwrap_or_default().to_string();
            node.systems = e.systems.clone();
            node.flow = e.flow;
        }
        let mut size = None;
        match e.shape {
            Some(Shape::Axis { start, end, width, height }) if node.kind == NodeKind::DuctNoGeometry => {
                node.kind = NodeKind::Duct { start, end, width, height };
                report.geometry += 1;
            }
            Some(Shape::Box { frame, min, max }) => {
                node.location.get_or_insert(frame.apply(mul(add(min, max), 0.5)));
                size = Some(sub(max, min));
                boxes.push((i, frame, min, max));
            }
            _ => {}
        }
        node.ifc = Some(IfcInfo {
            class: e.class.clone(),
            predefined: e.predefined.clone(),
            description: e.description.clone(),
            size,
        });
    }
    for (k, e) in elements.iter().enumerate() {
        for &j in &e.joined {
            report.ports += net.connect(nodes[k], nodes[j], EdgeSource::Recorded) as usize;
        }
    }

    let joined: HashSet<usize> = net.edges.iter().flat_map(|e| [e.a, e.b]).collect();
    let ends: Vec<(usize, Point)> = net
        .nodes
        .iter()
        .enumerate()
        .filter_map(|(i, n)| match n.kind {
            NodeKind::Duct { start, end, .. } => Some([(i, start), (i, end)]),
            _ => None,
        })
        .flatten()
        .collect();
    for (i, frame, min, max) in boxes.into_iter().filter(|(i, ..)| !joined.contains(i)) {
        for &(duct, p) in &ends {
            let q = frame.local(p);
            if (0..3).all(|k| q[k] > min[k] - INSIDE_TOL && q[k] < max[k] + INSIDE_TOL)
                && net.connect(i, duct, EdgeSource::EndInside)
            {
                report.joined += 1;
            }
        }
    }
    report
}

#[cfg(test)]
mod tests {
    use super::*;

    const DUCT: &str = "ISO-10303-21;\nDATA;\n\
        #1=IFCCARTESIANPOINT((0.,0.,0.));\n#2=IFCDIRECTION((1.,0.,0.));\n#3=IFCDIRECTION((0.,1.,0.));\n#4=IFCDIRECTION((0.,0.,1.));\n\
        #5=IFCSIUNIT(*,.LENGTHUNIT.,.MILLI.,.METRE.);\n#6=IFCUNITASSIGNMENT((#5));\n\
        #10=IFCCARTESIANPOINT((1000.,2000.,3000.));\n#11=IFCAXIS2PLACEMENT3D(#10,#2,#4);\n#12=IFCLOCALPLACEMENT($,#11);\n\
        #20=IFCCARTESIANPOINTLIST2D(((-100.,0.),(0.,100.),(100.,0.),(0.,-100.)));\n\
        #21=IFCINDEXEDPOLYCURVE(#20,(IFCARCINDEX((1,2,3)),IFCARCINDEX((3,4,1))),$);\n\
        #22=IFCARBITRARYCLOSEDPROFILEDEF(.AREA.,$,#21);\n#23=IFCAXIS2PLACEMENT3D(#1,#2,#3);\n\
        #24=IFCEXTRUDEDAREASOLID(#22,#23,#4,500.);\n\
        #25=IFCSHAPEREPRESENTATION($,'Body','SweptSolid',(#24));\n#26=IFCPRODUCTDEFINITIONSHAPE($,$,(#25));\n\
        #30=IFCDUCTSEGMENT('g',$,'Py\\X\\F6re\\X\\E4 kanava','d',$,#12,#26,'42',.RIGIDSEGMENT.);\nENDSEC;";

    #[test]
    fn duct_axis_from_placement_and_extrusion() {
        let e = &parse(DUCT.as_bytes())[0];
        assert_eq!((e.tag, e.class.as_str(), e.predefined.as_str(), e.name.as_str()), (Some(42), "IFCDUCTSEGMENT", "RIGIDSEGMENT", "Pyöreä kanava"));
        // The element's x axis is its RefDirection, world z; the extrusion runs
        // along its position's z, which is the element's x: so up, 500 mm.
        let Some(Shape::Axis { start, end, width, height }) = e.shape else { panic!("no axis") };
        let close = |a: Point, b: Point| (0..3).all(|k| (a[k] - b[k]).abs() < 1e-9);
        assert!(close(start, [1.0, 2.0, 3.0]) && close(end, [1.0, 2.0, 3.5]), "{start:?} -> {end:?}");
        assert!((width - 0.2).abs() < 1e-9 && (height - 0.2).abs() < 1e-9);
    }

    /// A duct and a terminal joined port to port (IFC2x3 style), the
    /// terminal's flow and system in a Revit-style property set.
    const PORTS: &str = "ISO-10303-21;\nDATA;\n\
        #1=IFCDUCTSEGMENT('a',$,'Rectangular Duct:Supply:7',$,$,$,$,'7',$);\n\
        #2=IFCAIRTERMINAL('b',$,'M_Supply Diffuser:600 x 600:8',$,$,$,$,'8',$);\n\
        #3=IFCDISTRIBUTIONPORT('c',$,$,$,$,$,$,.SOURCE.);\n#4=IFCDISTRIBUTIONPORT('d',$,$,$,$,$,$,.SINK.);\n\
        #5=IFCRELCONNECTSPORTTOELEMENT('e',$,$,$,#3,#1);\n#6=IFCRELCONNECTSPORTTOELEMENT('f',$,$,$,#4,#2);\n\
        #7=IFCRELCONNECTSPORTS('g',$,$,$,#3,#4,$);\n\
        #8=IFCPROPERTYSINGLEVALUE('Flow',$,IFCREAL(1.),$);\n\
        #9=IFCPROPERTYSINGLEVALUE('System Name',$,IFCLABEL('Mechanical Supply Air 1'),$);\n\
        #10=IFCPROPERTYSET('h',$,'Mechanical',$,(#8,#9));\n#11=IFCRELDEFINESBYPROPERTIES('i',$,$,$,(#2),#10);\nENDSEC;";

    #[test]
    fn ports_and_properties_make_an_ifc_only_network() {
        let elements = parse(PORTS.as_bytes());
        let terminal = &elements[1];
        assert_eq!(terminal.systems, ["Mechanical Supply Air 1"]);
        assert!((terminal.flow.unwrap() - CFS).abs() < 1e-12);

        let mut net = Network::default();
        let report = merge(&mut net, &elements);
        assert_eq!((report.added, report.ports), (2, 1));
        let t = &net.nodes[net.find(8).unwrap()];
        assert_eq!((t.kind.clone(), t.family.as_str(), t.type_name.as_str()), (NodeKind::Terminal, "M_Supply Diffuser", "600 x 600"));
        assert_eq!(net.neighbours()[net.find(7).unwrap()], [net.find(8).unwrap()]);
    }

    /// MagiCAD's export: flows in L/s, named in the model's language, and a
    /// system group in place of a "System Name".
    const MAGICAD: &str = "ISO-10303-21;\nDATA;\n\
        #1=IFCUNITASSIGNMENT((#2,#4));\n#2=IFCSIUNIT(*,.VOLUMEUNIT.,.DECI.,.CUBIC_METRE.);\n\
        #3=IFCSIUNIT(*,.TIMEUNIT.,$,.SECOND.);\n#4=IFCDERIVEDUNIT((#5,#6),.VOLUMETRICFLOWRATEUNIT.,$);\n\
        #5=IFCDERIVEDUNITELEMENT(#2,1);\n#6=IFCDERIVEDUNITELEMENT(#3,-1);\n\
        #7=IFCAIRTERMINAL('a',$,'CLIK-125',$,$,$,$,'8',$);\n\
        #8=IFCPROPERTYSINGLEVALUE('Virtaama',$,IFCVOLUMETRICFLOWRATEMEASURE(15.0000),$);\n\
        #9=IFCPROPERTYSINGLEVALUE('Virtaama, todellinen',$,IFCVOLUMETRICFLOWRATEMEASURE(0.0000),$);\n\
        #10=IFCPROPERTYSET('b',$,'FI_Tekninen',$,(#9,#8));\n#11=IFCRELDEFINESBYPROPERTIES('c',$,$,$,(#7),#10);\n\
        #12=IFCDISTRIBUTIONSYSTEM('d',$,'Tuloilma (T1)','T1',$,$,.NOTDEFINED.);\n\
        #13=IFCRELASSIGNSTOGROUP('e',$,$,$,(#7),$,#12);\nENDSEC;";

    #[test]
    fn typed_flow_in_the_project_unit_and_system_groups() {
        let terminal = &parse(MAGICAD.as_bytes())[0];
        assert_eq!(terminal.systems, ["Tuloilma (T1)"]);
        assert!((terminal.flow.unwrap() - 0.015).abs() < 1e-12);
    }
}
