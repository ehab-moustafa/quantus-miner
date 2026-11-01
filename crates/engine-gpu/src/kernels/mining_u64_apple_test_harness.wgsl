// ---------------------------------------------------------------------------
// Compatibility layer: same API as mining.wgsl, backed by the u64 core above.
// Only used by the component test harness; the mining kernel never calls it.
// All outputs are canonical, matching the reference implementation.
// ---------------------------------------------------------------------------

struct GoldilocksField {
    limb0: u32,
    limb1: u32,
}

const INTERNAL_CONSTANTS: array<array<u32, 2>, 22> = array<array<u32, 2>, 22>(
    array<u32, 2>(2018170979u, 2549578122u),
    array<u32, 2>(794875120u, 3520249608u),
    array<u32, 2>(2677723654u, 1772320679u),
    array<u32, 2>(2743438884u, 2849007878u),
    array<u32, 2>(518907317u, 693269760u),
    array<u32, 2>(293328710u, 1484055617u),
    array<u32, 2>(2834138828u, 2315799483u),
    array<u32, 2>(1558078501u, 1039128420u),
    array<u32, 2>(2266808631u, 966316006u),
    array<u32, 2>(3393728842u, 1045622667u),
    array<u32, 2>(2245828300u, 2521440415u),
    array<u32, 2>(751064958u, 1070374632u),
    array<u32, 2>(3490278765u, 2390340773u),
    array<u32, 2>(3526960470u, 2224174634u),
    array<u32, 2>(639988950u, 4000511088u),
    array<u32, 2>(1839350858u, 504240201u),
    array<u32, 2>(559852230u, 255489215u),
    array<u32, 2>(2713771731u, 453385078u),
    array<u32, 2>(1745082278u, 422331096u),
    array<u32, 2>(2544763488u, 4141129721u),
    array<u32, 2>(2700752774u, 1052996327u),
    array<u32, 2>(4063512019u, 1429786100u)
);

const INITIAL_EXTERNAL_CONSTANTS: array<array<array<u32, 2>, 12>, 4> = array<array<array<u32, 2>, 12>, 4>(
    array<array<u32, 2>, 12>(
        array<u32, 2>(2539329031u, 3221415792u),
        array<u32, 2>(4262746426u, 3164936845u),
        array<u32, 2>(3883202553u, 1922272763u),
        array<u32, 2>(3761386668u, 3841130025u),
        array<u32, 2>(1411081289u, 3588274735u),
        array<u32, 2>(4090250945u, 3962812520u),
        array<u32, 2>(1100826458u, 1215155029u),
        array<u32, 2>(1489773809u, 1813820067u),
        array<u32, 2>(2585015995u, 3824356688u),
        array<u32, 2>(2378857513u, 3651555078u),
        array<u32, 2>(2864423342u, 3852156759u),
        array<u32, 2>(1531416540u, 708695120u)
    ),
    array<array<u32, 2>, 12>(
        array<u32, 2>(1987505445u, 2913073372u),
        array<u32, 2>(1426707734u, 655469195u),
        array<u32, 2>(3385403543u, 1256631504u),
        array<u32, 2>(1381422714u, 1458257259u),
        array<u32, 2>(2453402910u, 528129365u),
        array<u32, 2>(964329320u, 905986685u),
        array<u32, 2>(1534247888u, 3842469367u),
        array<u32, 2>(744525997u, 4241857185u),
        array<u32, 2>(1756723870u, 3448331916u),
        array<u32, 2>(3610291774u, 1105166073u),
        array<u32, 2>(2596181885u, 3997051784u),
        array<u32, 2>(3199845381u, 3533420525u)
    ),
    array<array<u32, 2>, 12>(
        array<u32, 2>(4127777666u, 2103183598u),
        array<u32, 2>(2867888172u, 2309916828u),
        array<u32, 2>(1831532055u, 3009056407u),
        array<u32, 2>(2947502451u, 3675530062u),
        array<u32, 2>(3565886616u, 2029012066u),
        array<u32, 2>(3833391242u, 642945968u),
        array<u32, 2>(1773785903u, 2577032347u),
        array<u32, 2>(1770914259u, 1689297286u),
        array<u32, 2>(3752758200u, 3993707216u),
        array<u32, 2>(3389302766u, 1339375184u),
        array<u32, 2>(2180141127u, 1466089441u),
        array<u32, 2>(3199591357u, 4111832034u)
    ),
    array<array<u32, 2>, 12>(
        array<u32, 2>(1625498743u, 509487959u),
        array<u32, 2>(4188712685u, 1646551713u),
        array<u32, 2>(3451003566u, 2854767422u),
        array<u32, 2>(1412166652u, 1674110767u),
        array<u32, 2>(3410212320u, 1000704202u),
        array<u32, 2>(3381743837u, 602777331u),
        array<u32, 2>(3131873882u, 2866003620u),
        array<u32, 2>(2610174026u, 3923414377u),
        array<u32, 2>(3644719692u, 3450945356u),
        array<u32, 2>(1458984419u, 2418851081u),
        array<u32, 2>(3344519983u, 1531855103u),
        array<u32, 2>(2721413879u, 3732495392u)
    )
);

const TERMINAL_EXTERNAL_CONSTANTS: array<array<array<u32, 2>, 12>, 4> = array<array<array<u32, 2>, 12>, 4>(
    array<array<u32, 2>, 12>(
        array<u32, 2>(3773451374u, 2071119161u),
        array<u32, 2>(3805190518u, 340095962u),
        array<u32, 2>(2402679944u, 2149591222u),
        array<u32, 2>(743434178u, 1832305922u),
        array<u32, 2>(2847530739u, 2718290175u),
        array<u32, 2>(514243119u, 4142392203u),
        array<u32, 2>(3844443492u, 888639642u),
        array<u32, 2>(2008645578u, 2957397405u),
        array<u32, 2>(3732799654u, 1692252629u),
        array<u32, 2>(664231319u, 248567644u),
        array<u32, 2>(287781771u, 482031345u),
        array<u32, 2>(3486561978u, 1718871301u)
    ),
    array<array<u32, 2>, 12>(
        array<u32, 2>(814165505u, 1616875560u),
        array<u32, 2>(2123759183u, 1070984082u),
        array<u32, 2>(2722916813u, 3893372341u),
        array<u32, 2>(3726899022u, 4157656693u),
        array<u32, 2>(2824360073u, 4086907574u),
        array<u32, 2>(4155973110u, 1837140488u),
        array<u32, 2>(2297731723u, 4169165669u),
        array<u32, 2>(707924090u, 1474243980u),
        array<u32, 2>(1298483757u, 384287239u),
        array<u32, 2>(4243798069u, 557703745u),
        array<u32, 2>(1510569718u, 2968696976u),
        array<u32, 2>(3174388759u, 3638808363u)
    ),
    array<array<u32, 2>, 12>(
        array<u32, 2>(2044277519u, 3835193622u),
        array<u32, 2>(2743212840u, 1983595986u),
        array<u32, 2>(3943309069u, 151568229u),
        array<u32, 2>(424355060u, 1989274413u),
        array<u32, 2>(867046322u, 239293714u),
        array<u32, 2>(4230997871u, 2479068123u),
        array<u32, 2>(1565052394u, 2566260552u),
        array<u32, 2>(815274432u, 3822673712u),
        array<u32, 2>(1051683535u, 519405993u),
        array<u32, 2>(2687564964u, 186958263u),
        array<u32, 2>(1450226471u, 1648586942u),
        array<u32, 2>(1511122054u, 1595811937u)
    ),
    array<array<u32, 2>, 12>(
        array<u32, 2>(2426274859u, 4261676319u),
        array<u32, 2>(1207777283u, 1918363057u),
        array<u32, 2>(3090099832u, 3870167883u),
        array<u32, 2>(4101522705u, 1460796764u),
        array<u32, 2>(201900220u, 4164567654u),
        array<u32, 2>(2587682901u, 752404845u),
        array<u32, 2>(2967564913u, 2100296475u),
        array<u32, 2>(3404347409u, 2242778408u),
        array<u32, 2>(3350048952u, 1386431957u),
        array<u32, 2>(4093308564u, 1347177553u),
        array<u32, 2>(2633812729u, 3169012324u),
        array<u32, 2>(1727753673u, 3768793234u)
    )
);

fn gf_pack(g: GoldilocksField) -> u64 {
    return (u64(g.limb1) << 32u) | u64(g.limb0);
}

fn gf_unpack(v: u64) -> GoldilocksField {
    let c = gf64_canon(v);
    return GoldilocksField(u32(c & EPS64), u32(c >> 32u));
}

fn gf_from_limbs(l0: u32, l1: u32) -> GoldilocksField {
    return GoldilocksField(l0, l1);
}

fn gf_zero() -> GoldilocksField {
    return GoldilocksField(0u, 0u);
}

fn gf_one() -> GoldilocksField {
    return GoldilocksField(1u, 0u);
}

fn gf_from_u32(val: u32) -> GoldilocksField {
    return GoldilocksField(val, 0u);
}

fn gf_from_u64_parts(low: u32, high: u32) -> GoldilocksField {
    return gf_unpack((u64(high) << 32u) | u64(low));
}

fn gf_from_const(val: array<u32, 2>) -> GoldilocksField {
    return gf_from_u64_parts(val[0], val[1]);
}

fn gf_add(a: GoldilocksField, b: GoldilocksField) -> GoldilocksField {
    return gf_unpack(gf64_add(gf_pack(a), gf_pack(b)));
}

fn gf_mul(a: GoldilocksField, b: GoldilocksField) -> GoldilocksField {
    return gf_unpack(gf64_mul(gf_pack(a), gf_pack(b)));
}

fn sbox(x: GoldilocksField) -> GoldilocksField {
    return gf_unpack(gf64_sbox(gf_pack(x)));
}

fn state_pack(state: ptr<function, array<GoldilocksField, 12>>, out: ptr<function, array<u64, 12>>) {
    for (var i = 0u; i < 12u; i++) {
        (*out)[i] = gf_pack((*state)[i]);
    }
}

fn state_unpack(v: ptr<function, array<u64, 12>>, state: ptr<function, array<GoldilocksField, 12>>) {
    for (var i = 0u; i < 12u; i++) {
        (*state)[i] = gf_unpack((*v)[i]);
    }
}

fn external_linear_layer(state: ptr<function, array<GoldilocksField, 12>>) {
    var st: array<u64, 12>;
    state_pack(state, &st);
    ext_layer64(&st, 8u);
    state_unpack(&st, state);
}

fn internal_linear_layer(state: ptr<function, array<GoldilocksField, 12>>) {
    var st: array<u64, 12>;
    state_pack(state, &st);
    int_layer64(&st, 0lu);
    state_unpack(&st, state);
}

fn poseidon2_permute(state: ptr<function, array<GoldilocksField, 12>>) {
    var st: array<u64, 12>;
    state_pack(state, &st);
    permute64(&st);
    state_unpack(&st, state);
}

fn bytes_to_field_elements(input: array<u32, 24>) -> array<GoldilocksField, 25> {
    var felts: array<GoldilocksField, 25>;
    for (var i = 0u; i < 24u; i++) {
        felts[i] = gf_from_u32(input[i]);
    }
    felts[24] = gf_one();
    return felts;
}

fn field_elements_to_bytes(felts: array<GoldilocksField, 4>) -> array<u32, 8> {
    var result: array<u32, 8>;
    for (var i = 0u; i < 4u; i++) {
        result[i * 2u] = felts[i].limb0;
        result[i * 2u + 1u] = felts[i].limb1;
    }
    return result;
}

fn poseidon2_hash_squeeze_twice(input: array<u32, 24>) -> array<u32, 16> {
    var st: array<u64, 12>;
    for (var i = 0u; i < 12u; i++) {
        st[i] = 0lu;
    }
    for (var chunk = 0u; chunk < 3u; chunk++) {
        for (var i = 0u; i < 8u; i++) {
            st[i] = gf64_add(st[i], u64(input[chunk * 8u + i]));
        }
        permute64(&st);
    }
    st[0] = gf64_add(st[0], 1lu);
    st[1] = gf64_add(st[1], 1lu);
    permute64(&st);

    var result: array<u32, 16>;
    for (var i = 0u; i < 4u; i++) {
        let c = gf64_canon(st[i]);
        result[2u * i] = u32(c & EPS64);
        result[2u * i + 1u] = u32(c >> 32u);
    }
    permute64(&st);
    for (var i = 0u; i < 4u; i++) {
        let c = gf64_canon(st[i]);
        result[8u + 2u * i] = u32(c & EPS64);
        result[8u + 2u * i + 1u] = u32(c >> 32u);
    }
    return result;
}

fn hash_squeeze_twice(input: array<u32, 24>) -> array<u32, 16> {
    return poseidon2_hash_squeeze_twice(input);
}
