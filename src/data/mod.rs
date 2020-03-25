#![deny(clippy::all)]

mod airports;
mod routes;

mod part0;
use part0::COLS;
use part0::DATA;
use part0::ROWS;
use part0::STEP;

#[derive(Debug)]
pub struct Index(usize, usize);

#[derive(Debug)]
pub struct Coordinate(f64, f64);

fn to_index(coord: &Coordinate) -> Index {
    let Coordinate(lat, lon) = coord;
    let row = (*lat * STEP as f64 + ROWS as f64 / 2.) as usize;
    let col = (*lon * STEP as f64 + COLS as f64 / 2.) as usize;
    Index(row, col)
}

fn to_coordinate(index: &Index) -> Coordinate {
    let Index(row, col) = index;
    let lat = (ROWS as f64 / 2. - *row as f64) / STEP as f64;
    let lon = (*col as f64 - COLS as f64 / 2.) / STEP as f64;
    Coordinate(lat, lon)
}

fn wrap_deg(deg: f64, is_lon: bool) -> f64 {
    if deg == 0. {
        return deg;
    }
    let range = if is_lon { 180. } else { 90. };
    if deg > 0. {
        let mod_deg = deg % (range * 2.);
        // wraps into neg degrees
        if mod_deg > range {
            return -range + (mod_deg - range);
        }
        return mod_deg;
    }
    // neg degrees
    -wrap_deg(-deg, is_lon)
}

impl Coordinate {
    pub fn new(lat: f64, lon: f64) -> Self {
        Coordinate(wrap_deg(lat, false), wrap_deg(lon, true))
    }

    pub fn range(start: &Coordinate, stop: &Coordinate) -> impl Iterator<Item = Coordinate> {
        let Index(row_start, col_start) = to_index(start);
        let Index(row_stop, col_stop) = to_index(stop);
        (row_start..=row_stop).flat_map(move |row| {
            (col_start..=col_stop).map(move |col| to_coordinate(&Index(row, col)))
        })
    }
}

pub fn enumerate() -> impl Iterator<Item = (Coordinate, &'static f64)> {
    DATA.iter().enumerate().flat_map(|(row_idx, row)| {
        row.iter()
            .enumerate()
            .filter(|(_col_idx, value)| **value > 0.)
            .map(move |(col_idx, value)| (to_coordinate(&Index(row_idx, col_idx)), value))
    })
}

pub fn values() -> impl Iterator<Item = &'static f64> {
    enumerate().map(|(_coord, value)| value)
}

pub fn vec3() -> impl Iterator<Item = (f64, f64, &'static f64)> {
    enumerate().map(|(Coordinate(lat, lon), value)| (lat, lon, value))
}

pub fn flat_vec3() -> impl Iterator<Item = f64> {
    vec3().flat_map(|(lat, lon, value)| vec![lat, lon, value.clone()])
}

pub fn lowp_flat_vec3() -> impl Iterator<Item = f32> {
    flat_vec3().map(|value| value as f32)
}

pub fn at(coord: &Coordinate) -> f64 {
    let Index(row, col) = to_index(coord);
    DATA[row][col]
}

// todo: and indexed vertex array would be better
pub fn routes() -> impl Iterator<Item = (Coordinate, Coordinate)> {
    routes::ROUTES
        .iter()
        .filter(|(id_start, id_stop)| {
            airports::AIRPORTS.get(*id_start).is_some()
                && airports::AIRPORTS.get(*id_stop).is_some()
        })
        .map(|(id_start, id_stop)| {
            let (_a, (_, _, lat_start, lon_start)) = airports::AIRPORTS.get(*id_start).unwrap();
            let (_a, (_, _, lat_stop, lon_stop)) = airports::AIRPORTS.get(*id_stop).unwrap();
            (
                Coordinate(*lat_start, *lon_start),
                Coordinate(*lat_stop, *lon_stop),
            )
        })
    // .filter(|(coord_start, coord_stop)| at(coord_start) > 200. || at(coord_stop) > 200.)
}

pub fn flat_routes_vec3() -> impl Iterator<Item = f64> {
    routes().flat_map(
        |(Coordinate(lat_start, lon_start), Coordinate(lat_stop, lon_stop))| {
            vec![
                lat_start, lon_start, 0., lat_stop, lon_stop, 1., lat_start, lon_start, 2.,
                lat_start, lon_start, 3., lat_stop, lon_stop, 4., lat_stop, lon_stop, 5.,
            ]
        },
    )
}

pub fn lowp_flat_routes_vec3() -> impl Iterator<Item = f32> {
    flat_routes_vec3().map(|value| value as f32)
}
