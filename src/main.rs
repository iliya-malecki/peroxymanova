use peroxymanova::{generate_data, _permanova};
fn main() {
    let (sqdistances, labels) = generate_data(600, 3);
    let (a, b) = _permanova(&sqdistances.view(), labels);
    dbg!(a);
    dbg!(b);
}
