use peroxymanova::{generate_data, core_permanova};
fn main() {
    let (sqdistances, labels) = generate_data(600, 3);
    let (a, b) = core_permanova(&sqdistances.view(), labels);
    dbg!(a);
    dbg!(b);
}
