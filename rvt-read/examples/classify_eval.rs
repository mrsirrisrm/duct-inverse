//! Score the family-name rules on eval sets: how many they answer, how many
//! of those are right, and every wrong or unanswered item.
use duct_heal::classify::{classify, Category};

fn main() {
    for path in std::env::args().skip(1) {
        let set: serde_json::Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        let (mut answered, mut right, mut missed) = (0, 0, Vec::new());
        let items = set["items"].as_array().unwrap();
        for item in items {
            let category = match item["category"].as_str().unwrap() {
                "Duct Fittings" | "Duct Accessories" => Category::Fitting,
                "Air Terminals" => Category::Terminal,
                "Mechanical Equipment" => Category::Equipment,
                _ => Category::Other,
            };
            let (family, ty) = (item["family"].as_str().unwrap(), item["type"].as_str().unwrap());
            let accept: Vec<&str> = item["accept"].as_array().unwrap().iter().map(|a| a.as_str().unwrap()).collect();
            match classify(category, family, ty).map(|t| t.label()) {
                Some(label) => {
                    answered += 1;
                    if accept.contains(&label) {
                        right += 1;
                    } else {
                        println!("  WRONG {family} | {ty}: {label}, want {accept:?}");
                    }
                }
                None => missed.push(format!("  ask   {family} | {ty} (want {accept:?})")),
            }
        }
        missed.iter().for_each(|m| println!("{m}"));
        println!("{path}: {} items, {answered} answered ({right} right), {} left for the classifier\n", items.len(), missed.len());
    }
}
