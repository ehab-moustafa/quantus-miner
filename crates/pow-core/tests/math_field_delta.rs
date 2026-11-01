//! Abstract op counts: full coefficient / pre-S-box rebuild vs one-add incremental
//! recurrences (Goldilocks p = 2^64 - 2^32 + 1). Models match `engine-cuda` `gf64_add_model`.

const GOLDILOCKS_P: u128 = 0xffff_ffff_0000_0001;
const P64: u64 = 0xffff_ffff_0000_0001;
const EPS: u64 = 0xffff_ffff;
const STEPS: u32 = 1024;

fn gf64_add_model(a: u64, b: u64) -> u64 {
    let (sum, carry) = a.overflowing_add(b);
    if !carry {
        return sum;
    }
    sum.wrapping_add(EPS)
}

fn u_rebuild(pk0: u64, t: u32) -> u64 {
    let d = 35u64.wrapping_mul(t as u64);
    gf64_add_model(pk0, P64.wrapping_sub(d))
}

fn v_rebuild(pk1: u64, t: u32) -> u64 {
    let d = 35u64.wrapping_mul(t as u64);
    gf64_add_model(pk1, d)
}

#[derive(Default, Debug, Clone, Copy)]
struct Ops {
    imul: u64,
    iadd: u64,
    u64_add: u64,
    gf64_add: u64,
}

impl Ops {
    fn add_assign(&mut self, other: Ops) {
        self.imul += other.imul;
        self.iadd += other.iadd;
        self.u64_add += other.u64_add;
        self.gf64_add += other.gf64_add;
    }

    fn total(&self) -> u64 {
        self.imul + self.iadd + self.u64_add + self.gf64_add
    }
}

fn full_presbox_step() -> Ops {
    Ops {
        imul: 1,
        gf64_add: 2,
        ..Default::default()
    }
}

fn incr_presbox_step() -> Ops {
    Ops {
        iadd: 1,
        gf64_add: 2,
        ..Default::default()
    }
}

fn full_mining_inject_step() -> Ops {
    Ops {
        u64_add: 1 + 12, // nonce index + coeff ladder (see GF64_CHAIN_AUDIT)
        gf64_add: 12,
        ..Default::default()
    }
}

fn incr_mining_inject_step() -> Ops {
    Ops {
        gf64_add: 12,
        ..Default::default()
    }
}

#[test]
fn math_field_delta_counts_1024_steps() {
    let mut full = Ops::default();
    let mut incr = Ops::default();
    for _ in 0..STEPS {
        full.add_assign(full_presbox_step());
        full.add_assign(full_mining_inject_step());
        incr.add_assign(incr_presbox_step());
        incr.add_assign(incr_mining_inject_step());
    }

    let ratio_total = full.total() as f64 / incr.total() as f64;
    let ratio_inject = (STEPS as u64 * (1 + 12 + 12)) as f64 / (STEPS as u64 * 12) as f64;
    let ratio_presbox_imul = (STEPS as u64) as f64 / (STEPS as u64 - 1) as f64;

    eprintln!("=== math_field_delta {} steps ===", STEPS);
    eprintln!("full ops:  {:?}", full);
    eprintln!("incr ops:  {:?}", incr);
    eprintln!(
        "ratio total (rebuild/incr): {:.4} (full={}, incr={})",
        ratio_total,
        full.total(),
        incr.total()
    );
    eprintln!(
        "ratio mining_main inject only: {:.4}",
        ratio_inject
    );
    eprintln!(
        "ratio nonce-line presbox imul-only: {:.4}",
        ratio_presbox_imul
    );

    assert!((ratio_total - 28672.0 / 15360.0).abs() < 0.001);
    assert!((ratio_inject - 25.0 / 12.0).abs() < 0.01);
}

#[test]
fn nonce_line_uv_incremental_matches_rebuild_1024_steps() {
    let pk0 = 0x486dcb55419d475a_u64;
    let pk1 = 0x6c1cb2a358cc24f1_u64;
    let neg35 = (GOLDILOCKS_P - 35) as u64;
    let mut u = u_rebuild(pk0, 0);
    let mut v = v_rebuild(pk1, 0);
    for t in 0..STEPS {
        assert_eq!(u, u_rebuild(pk0, t), "u mismatch at t={t}");
        assert_eq!(v, v_rebuild(pk1, t), "v mismatch at t={t}");
        u = gf64_add_model(u, neg35);
        v = gf64_add_model(v, 35);
    }
}
