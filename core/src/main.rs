use peroxymanova_core::{generate_data, permanova};
fn main() {
    let (sqdistances, labels) = generate_data(600, 3);
    let (a, b) = permanova(&sqdistances.view(), labels);
    dbg!(a);
    dbg!(b);
}
