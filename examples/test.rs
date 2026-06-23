use std::time::Instant;

const N: usize = 512;
const BLOCK_SIZE: usize = 32;

fn main() {
    let mut qwe: Vec<u8> = Vec::new();
    qwe.push(2u8);
    assert!(qwe.as_ptr() % 2u8 == 0);

    //     let now = Instant::now();
    //
    //     let a: Vec<f32> = (0..N * N).map(|x| x as f32).collect();
    //     let b: Vec<f32> = (0..N * N).map(|x| x as f32).collect();
    //     let mut c = vec![0.0; N * N];
    //
    //     for dx in (0..N).step_by(BLOCK_SIZE) {
    //         for dy in (0..N).step_by(BLOCK_SIZE) {
    //             for x in 0..BLOCK_SIZE {
    //                 for y in 0..BLOCK_SIZE {
    //                     let mut acc = 0.0;
    //                     for k in 0..N {
    //                         acc += a[(dx + x) * N + k] * b[k * N + (dy + y)];
    //                     }
    //                     c[(x + dx) * N + (y + dy)] = acc;
    //                 }
    //             }
    //         }
    //     }
    //
    //     let gflop = (2 * N * N * N) as f64 * 1e-9;
    //     let s = now.elapsed().as_secs_f64() as f64;
    //     let gflops = gflop / s;
    //     println!("GFLOP/s: {}, {}", gflops, c[44444]);
}
