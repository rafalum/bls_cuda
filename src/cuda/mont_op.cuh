#include "common.h"
#include "ptx.cuh"

#define CUDA_CHECK(call) if((errorState=call)!=0) { cudaError("Call \"" #call "\" failed.", __FILE__, __LINE__); return errorState; }

// ----------------------------------------------------------------------------
// Montgomery operations
// Mostly taken from: github.com/matter-labs/era-bellmann
// ----------------------------------------------------------------------------

template <unsigned OPS_COUNT = UINT32_MAX, bool CARRY_IN = false, bool CARRY_OUT = false> struct carry_chain {
    unsigned index;

    constexpr DEVICE_INLINE carry_chain() : index(0) {}

    DEVICE_INLINE uint32_t add(const uint32_t x, const uint32_t y) {
        index++;
        if (index == 1 && OPS_COUNT == 1 && !CARRY_IN && !CARRY_OUT)
            return ptx::add(x, y);
        else if (index == 1 && !CARRY_IN)
            return ptx::add_cc(x, y);
        else if (index < OPS_COUNT || CARRY_OUT)
            return ptx::addc_cc(x, y);
        else
            return ptx::addc(x, y);
  }


    DEVICE_INLINE uint32_t sub(const uint32_t x, const uint32_t y) {
        index++;
        if (index == 1 && OPS_COUNT == 1 && !CARRY_IN && !CARRY_OUT)
            return ptx::sub(x, y);
        else if (index == 1 && !CARRY_IN)
            return ptx::sub_cc(x, y);
        else if (index < OPS_COUNT || CARRY_OUT)
            return ptx::subc_cc(x, y);
        else
            return ptx::subc(x, y);
    }

};

template <bool SUBTRACT, bool CARRY_OUT> static constexpr DEVICE_INLINE uint32_t add_sub_limbs(storage *rs, const storage *xs, const storage *ys) {
    const uint32_t *x = xs->limbs;
    const uint32_t *y = ys->limbs;
    uint32_t *r = rs->limbs;
    carry_chain<CARRY_OUT ? TLC + 1 : TLC> chain;
#pragma unroll
    for (unsigned i = 0; i < TLC; i++)
        r[i] = SUBTRACT ? chain.sub(x[i], y[i]) : chain.add(x[i], y[i]);
    if (!CARRY_OUT)
        return 0;
    return SUBTRACT ? chain.sub(0, 0) : chain.add(0, 0);
  }

  static DEVICE_INLINE storage sub(const storage *xs, const storage *ys) {

	storage ret;
	const storage modulus = MODULUS;

	uint32_t carry = add_sub_limbs<true, true>(&ret, xs, ys);
	if(carry != 0) {
		add_sub_limbs<false, false>(&ret, &ret, &modulus);
	}
	return ret;
}

static constexpr DEVICE_INLINE bool is_odd(const storage &xs) { 
    return xs.limbs[0] & 1; 
}

static constexpr DEVICE_INLINE bool is_even(const storage &xs) { 
    return ~xs.limbs[0] & 1; 
}

static constexpr DEVICE_INLINE bool lt(const storage &xs, const storage &ys) {
    storage dummy = {};
    uint32_t carry = add_sub_limbs<true, true>(&dummy, &xs, &ys);
    return carry;
}

static constexpr DEVICE_INLINE bool eq(const storage &xs, const storage &ys) {
    const uint32_t *x = xs.limbs;
    const uint32_t *y = ys.limbs;
    uint32_t limbs_or = x[0] ^ y[0];
#pragma unroll
    for (unsigned i = 1; i < TLC; i++)
        limbs_or |= x[i] ^ y[i];
    return limbs_or == 0;
}

static constexpr DEVICE_INLINE bool is_zero(const storage &xs) {
    const uint32_t *x = xs.limbs;
    uint32_t limbs_or = x[0];
#pragma unroll
    for (unsigned i = 1; i < TLC; i++)
        limbs_or |= x[i];
    return limbs_or == 0;
}

static constexpr DEVICE_INLINE storage div_by_2(const storage &xs) {
    const uint32_t *x = xs.limbs;
    storage ret = {};
    uint32_t *r = ret.limbs;
    constexpr storage modulus = MODULUS;
#pragma unroll
    for (unsigned i = 0; i < TLC - 1; i++)
        r[i] = __funnelshift_rc(x[i], x[i + 1], 1);
    r[TLC - 1] = x[TLC - 1] >> 1;

    storage rs = {};
  	return add_sub_limbs<true, true>(&rs, &ret, &modulus) ? ret : rs;
}

static constexpr DEVICE_INLINE storage inverse(const storage &xs) {
    if (is_zero(xs))
      return xs;
    constexpr storage one = {1};
    constexpr storage modulus = MODULUS;
    storage u = xs;
    storage v = modulus;
    storage b = RR;
    storage c = {};
    while (!eq(u, one) && !eq(v, one)) {
        while (is_even(u)) {
            u = div_by_2(u);
            if (is_odd(b))
            add_sub_limbs<false, false>(&b, &b, &modulus);
            b = div_by_2(b);
        }
        while (is_even(v)) {
            v = div_by_2(v);
            if (is_odd(c))
            add_sub_limbs<false, false>(&c, &c, &modulus);
            c = div_by_2(c);
        }
        if (lt(v, u)) {
            add_sub_limbs<true, false>(&u, &u, &v);
            b = sub(&b, &c);
        } 
        else {
            add_sub_limbs<true, false>(&v, &v, &u);
            c = sub(&c, &b);
        }
    }
    return eq(u, one) ? b : c;
}


static DEVICE_INLINE void mul_n(uint32_t *acc, const uint32_t *a, uint32_t bi, size_t n = TLC) { 
#pragma unroll
    for (size_t i = 0; i < n; i += 2) {
        acc[i] = ptx::mul_lo(a[i], bi);
        acc[i + 1] = ptx::mul_hi(a[i], bi);
    }
}

static DEVICE_INLINE void cmad_n(uint32_t *acc, const uint32_t *a, uint32_t bi, size_t n = TLC) {
    acc[0] = ptx::mad_lo_cc(a[0], bi, acc[0]);
    acc[1] = ptx::madc_hi_cc(a[0], bi, acc[1]);
#pragma unroll
    for (size_t i = 2; i < n; i += 2) {
        acc[i] = ptx::madc_lo_cc(a[i], bi, acc[i]);
        acc[i + 1] = ptx::madc_hi_cc(a[i], bi, acc[i + 1]);
    }
}

static DEVICE_INLINE void madc_n_rshift(uint32_t *odd, const uint32_t *a, uint32_t bi) {
    constexpr uint32_t n = TLC;
#pragma unroll
    for (size_t i = 0; i < n - 2; i += 2) {
        odd[i] = ptx::madc_lo_cc(a[i], bi, odd[i + 2]);
        odd[i + 1] = ptx::madc_hi_cc(a[i], bi, odd[i + 3]);
    }
    odd[n - 2] = ptx::madc_lo_cc(a[n - 2], bi, 0);
    odd[n - 1] = ptx::madc_hi(a[n - 2], bi, 0);
}


static DEVICE_INLINE void mad_n_redc(uint32_t *even, uint32_t *odd, const uint32_t *a, uint32_t bi, bool first = false) {
    constexpr uint32_t n = TLC;
    constexpr auto modulus = MODULUS;
    const uint32_t *const MOD = modulus.limbs;
    if (first) {
        mul_n(odd, a + 1, bi);
        mul_n(even, a, bi);
    } 
    else {
        even[0] = ptx::add_cc(even[0], odd[1]);
        madc_n_rshift(odd, a + 1, bi);
        cmad_n(even, a, bi);
        odd[n - 1] = ptx::addc(odd[n - 1], 0);
    }
    uint32_t mi = even[0] * INV;
    cmad_n(odd, MOD + 1, mi);
    cmad_n(even, MOD, mi);
    odd[n - 1] = ptx::addc(odd[n - 1], 0);
}

static DEVICE_INLINE storage montmul_raw(const uint32_t *a, const uint32_t *b) {

	storage ret;
    const storage modulus = MODULUS;

	uint32_t *even = ret.limbs;
	__align__(8) uint32_t odd[TLC + 1];
	size_t i;
	#pragma unroll
	for (i = 0; i < TLC; i += 2) {
		mad_n_redc(&even[0], &odd[0], a, b[i], i == 0);
		mad_n_redc(&odd[0], &even[0], a, b[i + 1]);
	}
	// merge |even| and |odd|
	even[0] = ptx::add_cc(even[0], odd[1]);
	#pragma unroll
	for (i = 1; i < TLC - 1; i++)
		even[i] = ptx::addc_cc(even[i], odd[i + 1]);
	even[i] = ptx::addc(even[i], 0);

	storage rs;
  	return add_sub_limbs<true, true>(&rs, &ret, &modulus) ? ret : rs;

}


static DEVICE_INLINE storage dbl(const storage *xs) {

	storage ret;
	const storage modulus = MODULUS;
	
	const uint32_t *x = xs->limbs;

    uint32_t *r = ret.limbs;
    r[0] = x[0] << 1;
	#pragma unroll
    for (unsigned i = 1; i < TLC; i++)
        r[i] = __funnelshift_r(x[i - 1], x[i], 31);

	storage rs;
  	return add_sub_limbs<true, true>(&rs, &ret, &modulus) ? ret : rs;
}

static DEVICE_INLINE storage sqr(const storage *xs) {
	// TODO: Implement fast squaring
	return montmul_raw(xs->limbs, xs->limbs);
}


static DEVICE_INLINE storage mul(const storage *a, const storage *b) {

	return montmul_raw(a->limbs, b->limbs);
}