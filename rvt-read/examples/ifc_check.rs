//! What `ifc::parse` makes of an IFC file: classes, shapes, one element.
use duct_heal::ifc::{parse, Shape};

fn main() {
    let mut args = std::env::args().skip(1);
    let path = args.next().expect("usage: ifc_check <file.ifc> [tag]");
    let start = std::time::Instant::now();
    let elements = parse(&std::fs::read(path).unwrap());
    eprintln!("parsed in {:.1} s", start.elapsed().as_secs_f64());
    let mut counts = std::collections::BTreeMap::new();
    for e in &elements {
        let shape = match e.shape {
            Some(Shape::Axis { .. }) => "axis",
            Some(Shape::Box { .. }) => "box",
            None => "none",
        };
        *counts.entry((e.class.as_str(), shape)).or_insert(0) += 1;
    }
    counts.iter().for_each(|((class, shape), n)| println!("{n:5} {class} {shape}"));
    if let Some(tag) = args.next().and_then(|t| t.parse().ok()) {
        println!("{:#?}", elements.iter().find(|e| e.tag == Some(tag)));
    }
}
