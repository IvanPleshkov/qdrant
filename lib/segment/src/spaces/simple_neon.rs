use crate::types::{ScoreType, VectorElementType};

pub fn has_neon() -> bool {
    true
}

pub unsafe fn euclid_similarity_neon(
    v1: &[VectorElementType],
    v2: &[VectorElementType],
) -> ScoreType {
    let mut result: f32 = 0.;
    let n = v1.len();
    let m = n - (n % 4);
    if m > 0 {
        std::arch::asm!(
            "mov {counter}, #0",
            "movi.2d v0, #0000000000000000",
            "1:",
            "cmp {counter}, {conter_end}",
            "beq 2f",
            "ldr q1, [{pointer1}], #16",
            "ldr q2, [{pointer2}], #16",
            "fsub.4s v3, v1, v2",
            "fmla.4s v0, v3, v3",
            "add {counter}, {counter}, #4",
            "b 1b",
            "2:",
            "faddp.4s v0, v0, v0",
            "faddp.2s s0, v0",
            "fmov {result:w}, s0",
            pointer1 = in(reg) v1.as_ptr(),
            pointer2 = in(reg) v2.as_ptr(),
            result = out(reg) result,
            counter = out(reg) _,
            conter_end = in(reg) m,
        );
    }

    for i in m..n {
        result += (v1[i] - v2[i]).powi(2);
    }
    -result.sqrt()
}

pub unsafe fn cosine_preprocess_neon(vector: &[VectorElementType]) -> Vec<VectorElementType> {
    let mut length: f32 = 0.;
    let n = vector.len();
    let m = n - (n % 4);
    if m > 0 {
        std::arch::asm!(
            "mov {counter}, #0",
            "movi.2d v0, #0000000000000000",
            "1:",
            "cmp {counter}, {conter_end}",
            "beq 2f",
            "ldr q1, [{pointer1}], #16",
            "fmla.4s v0, v1, v1",
            "add {counter}, {counter}, #4",
            "b 1b",
            "2:",
            "faddp.4s v0, v0, v0",
            "faddp.2s s0, v0",
            "fmov {result:w}, s0",
            pointer1 = in(reg) vector.as_ptr(),
            result = out(reg) length,
            counter = out(reg) _,
            conter_end = in(reg) m,
        );
    }

    for v in vector.iter().take(n).skip(m) {
        length += v.powi(2);
    }
    let length = length.sqrt();
    vector.iter().map(|x| x / length).collect()
}

pub unsafe fn dot_similarity_neon(v1: &[VectorElementType], v2: &[VectorElementType]) -> ScoreType {
    let mut result: f32 = 0.;
    let n = v1.len();
    let m = n - (n % 4);
    if m > 0 {
        std::arch::asm!(
            "mov {counter}, #0",
            "movi.2d v0, #0000000000000000",
            "1:",
            "cmp {counter}, {conter_end}",
            "beq 2f",
            "ldr q1, [{pointer1}], #16",
            "ldr q2, [{pointer2}], #16",
            "fmla.4s v0, v1, v2",
            "add {counter}, {counter}, #4",
            "b 1b",
            "2:",
            "faddp.4s v0, v0, v0",
            "faddp.2s s0, v0",
            "fmov {result:w}, s0",
            pointer1 = in(reg) v1.as_ptr(),
            pointer2 = in(reg) v2.as_ptr(),
            result = out(reg) result,
            counter = out(reg) _,
            conter_end = in(reg) m,
        );
    }

    for i in m..n {
        result += v1[i] * v2[i];
    }
    result
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_spaces_neon() {
        use super::*;
        use crate::spaces::simple::*;

        if has_neon() {
            let v1: Vec<f32> = vec![
                10.67, 11., 12., 13., 14., 15., 16., 17., 18., 19., 20., 21., 22., 23., 24., 25.,
                26., 27., 28., 29., 30., 31.,
            ];
            let v2: Vec<f32> = vec![
                40.34, 41., 42., 43., 44., 45., 46., 47., 48., 49., 50., 51., 52., 53., 54., 55.,
                56., 57., 58., 59., 60., 61.,
            ];

            /*
            let v1: Vec<f32> = vec![
                0., 1., 2.,
            ];
            let v2: Vec<f32> = vec![
                1., 1., 1.,
            ];
            */

            let euclid_simd = unsafe { euclid_similarity_neon(&v1, &v2) };
            let euclid = euclid_similarity(&v1, &v2);
            assert_eq!(euclid_simd, euclid);

            let dot_simd = unsafe { dot_similarity_neon(&v1, &v2) };
            let dot = dot_similarity(&v1, &v2);
            assert_eq!(dot_simd, dot);

            let cosine_simd = unsafe { cosine_preprocess_neon(&v1) };
            let cosine = cosine_preprocess(&v1);
            assert_eq!(cosine_simd, cosine);
        } else {
            println!("neon test skipped");
        }
    }
}
