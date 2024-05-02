#include <cstdint>

#include "cuda/kernels.cuh"

namespace host_math {

// return x + y with uint32_t operands
static __host__ uint32_t add(const uint32_t x, const uint32_t y) { return x + y; }

// return x + y + carry with uint32_t operands
static __host__ uint32_t addc(const uint32_t x, const uint32_t y, const uint32_t carry) { return x + y + carry; }

// return x + y and carry out with uint32_t operands
static __host__ uint32_t add_cc(const uint32_t x, const uint32_t y, uint32_t &carry) {
  uint32_t result;
  result = x + y;
  carry = x > result;
  return result;
}

// return x + y + carry and carry out  with uint32_t operands
static __host__ uint32_t addc_cc(const uint32_t x, const uint32_t y, uint32_t &carry) {
  const uint32_t result = x + y + carry;
  carry = carry && x >= result || !carry && x > result;
  return result;
}

// return x - y with uint32_t operands
static __host__ uint32_t sub(const uint32_t x, const uint32_t y) { return x - y; }

// 	return x - y - borrow with uint32_t operands
static __host__ uint32_t subc(const uint32_t x, const uint32_t y, const uint32_t borrow) { return x - y - borrow; }

//	return x - y and borrow out with uint32_t operands
static __host__ uint32_t sub_cc(const uint32_t x, const uint32_t y, uint32_t &borrow) {
  uint32_t result;
  result = x - y;
  borrow = x < result;
  return result;
}

//	return x - y - borrow and borrow out with uint32_t operands
static __host__ uint32_t subc_cc(const uint32_t x, const uint32_t y, uint32_t &borrow) {
  const uint32_t result = x - y - borrow;
  borrow = borrow && x <= result || !borrow && x < result;
  return result;
}

// return x * y + z + carry and carry out with uint32_t operands
static __host__ uint32_t madc_cc(const uint32_t x, const uint32_t y, const uint32_t z, uint32_t &carry) {
  uint32_t result;
  uint64_t r = static_cast<uint64_t>(x) * y + z + carry;
  carry = r >> 32;
  result = r & 0xffffffff;
  return result;
}

} // namespace host_math

// ----------------------------------------------------------------------------
// Montgomery operations on the host
// Mostly taken from: github.com/matter-labs/era-bellmann
// ----------------------------------------------------------------------------

namespace host {

template <unsigned OPS_COUNT = UINT32_MAX, bool CARRY_IN = false, bool CARRY_OUT = false> struct carry_chain {
  unsigned index;

  constexpr HOST_INLINE carry_chain() : index(0) {}

  HOST_INLINE uint32_t add(const uint32_t x, const uint32_t y, uint32_t &carry) {
    index++;
    if (index == 1 && OPS_COUNT == 1 && !CARRY_IN && !CARRY_OUT)
      return host_math::add(x, y);
    else if (index == 1 && !CARRY_IN)
      return host_math::add_cc(x, y, carry);
    else if (index < OPS_COUNT || CARRY_OUT)
      return host_math::addc_cc(x, y, carry);
    else
      return host_math::addc(x, y, carry);
  }

  HOST_INLINE uint32_t sub(const uint32_t x, const uint32_t y, uint32_t &carry) {
    index++;
    if (index == 1 && OPS_COUNT == 1 && !CARRY_IN && !CARRY_OUT)
      return host_math::sub(x, y);
    else if (index == 1 && !CARRY_IN)
      return host_math::sub_cc(x, y, carry);
    else if (index < OPS_COUNT || CARRY_OUT)
      return host_math::subc_cc(x, y, carry);
    else
      return host_math::subc(x, y, carry);
  }

};

static constexpr HOST_INLINE bool eq(const storage &xs, const storage &ys) {
    for (unsigned i = 0; i < TLC; i++)
      if (xs.limbs[i] != ys.limbs[i])
        return false;
    return true;
  }

  static constexpr HOST_INLINE bool is_zero(const storage &xs) {
    for (unsigned i = 0; i < TLC; i++)
      if (xs.limbs[i])
        return false;
    return true;
  }
  
  template <bool SUBTRACT, bool CARRY_OUT> 
  static constexpr HOST_INLINE uint32_t add_sub_limbs(storage *rs, const storage *xs, const storage *ys) {
    const uint32_t *x = xs->limbs;
    const uint32_t *y = ys->limbs;
    uint32_t *r = rs->limbs;
    uint32_t carry = 0;
    host::carry_chain<TLC, false, CARRY_OUT> chain;
    for (unsigned i = 0; i < TLC; i++)
      r[i] = SUBTRACT ? chain.sub(x[i], y[i], carry) : chain.add(x[i], y[i], carry);
    return CARRY_OUT ? carry : 0;
  }

  static HOST_INLINE storage dbl(const storage *xs) {

    storage ret;
    const storage modulus = MODULUS;

    const uint32_t *x = xs->limbs;

    uint32_t *r = ret.limbs;
    r[0] = x[0] << 1;
    for (unsigned i = 1; i < TLC; i++)
      r[i] = (x[i] << 1) | (x[i - 1] >> 31);
    
    storage rs;
    return host::add_sub_limbs<true, true>(&rs, &ret, &modulus) ? ret : rs;
  }

  static HOST_INLINE storage div_by_2(const storage &xs) {
    const uint32_t *x = xs.limbs;
    storage ret = {};
    uint32_t *r = ret.limbs;
    constexpr storage modulus = MODULUS;
    for (unsigned i = 0; i < TLC - 1; i++)
      r[i] = (x[i] >> 1) | (x[i + 1] << 31);
    r[TLC - 1] = x[TLC - 1] >> 1;
    
    storage rs = {};
  	return host::add_sub_limbs<true, true>(&rs, &ret, &modulus) ? ret : rs;
  }

  static HOST_INLINE storage sub(const storage *xs, const storage *ys) {

    storage ret;
    const storage modulus = MODULUS;

    uint32_t carry = host::add_sub_limbs<true, true>(&ret, xs, ys);
    if(carry != 0) {
        host::add_sub_limbs<false, false>(&ret, &ret, &modulus);
    }
    return ret;
}

  static constexpr HOST_INLINE bool is_odd(const storage &xs) { return xs.limbs[0] & 1; }

  static constexpr HOST_INLINE bool is_even(const storage &xs) { return ~xs.limbs[0] & 1; }

  static constexpr HOST_INLINE bool lt(const storage &xs, const storage &ys) {
    storage dummy = {};
    uint32_t carry = host::add_sub_limbs<true, true>(&dummy, &xs, &ys);
    return carry;
  }

  static constexpr HOST_INLINE storage inverse(const storage &xs) {
    if (host::is_zero(xs))
      return xs;
    constexpr storage one = {1};
    constexpr storage modulus = MODULUS;
    storage u = xs;
    storage v = modulus;
    storage b = RR;
    storage c = {};
    while (!host::eq(u, one) && !host::eq(v, one)) {
      while (host::is_even(u)) {
        u = host::div_by_2(u);
        if (host::is_odd(b))
          host::add_sub_limbs<false, false>(&b, &b, &modulus);
        b = host::div_by_2(b);
      }
      while (host::is_even(v)) {
        v = host::div_by_2(v);
        if (host::is_odd(c))
          host::add_sub_limbs<false, false>(&c, &c, &modulus);
        c = host::div_by_2(c);
      }
      if (host::lt(v, u)) {
        host::add_sub_limbs<true, false>(&u, &u, &v);
        b = host::sub(&b, &c);
      } else {
        host::add_sub_limbs<true, false>(&v, &v, &u);
        c = host::sub(&c, &b);
      }
    }
    return host::eq(u, one) ? b : c;
  }

  static constexpr HOST_INLINE storage mul(const storage *xs, const storage *ys) {

    const uint32_t *x = xs->limbs;
    constexpr storage modulus = MODULUS;
    const uint32_t *const n = modulus.limbs;
    constexpr uint32_t q = INV;
    uint32_t t[TLC + 2] = {};
    for (const uint32_t y_limb : ys->limbs) {
      uint32_t carry = 0;
      for (unsigned i = 0; i < TLC; i++)
        t[i] = host_math::madc_cc(x[i], y_limb, t[i], carry);
      t[TLC] = host_math::add_cc(t[TLC], carry, carry);
      t[TLC + 1] = carry;
      carry = 0;
      const uint32_t m = q * t[0];
      host_math::madc_cc(m, n[0], t[0], carry);
      for (unsigned i = 1; i < TLC; i++)
        t[i - 1] = host_math::madc_cc(m, n[i], t[i], carry);
      t[TLC - 1] = host_math::add_cc(t[TLC], carry, carry);
      t[TLC] = t[TLC + 1] + carry;
    }
    storage rs = {};
    const storage ret = *reinterpret_cast<storage *>(t);
    return host::add_sub_limbs<true, true>(&rs, &ret, &modulus) ? ret : rs;
  }


  static constexpr HOST_INLINE storage sqr(const storage *xs) {
      return host::mul(xs, xs);
  }

  // https://hyperelliptic.org/EFD/g1p/auto-shortw-xyzz.html#addition-add-2008-s
static HOST_INLINE point_xyzz add(const point_xyzz *p1, const point_xyzz *p2) {
    
    const storage X1 = p1->x;     // < 2
    const storage Y1 = p1->y;     // < 2
    const storage ZZ1 = p1->zz;   // < 2
    const storage ZZZ1 = p1->zzz; // < 2
    const storage X2 = p2->x;     // < 2
    const storage Y2 = p2->y;     // < 2
    const storage ZZ2 = p2->zz;   // < 2
    const storage ZZZ2 = p2->zzz; // < 2

    const storage U1 = host::mul(&X1, &ZZ2);      // U1 = X1*ZZ2   < 2
    const storage U2 = host::mul(&X2, &ZZ1);      // U2 = X2*ZZ1   < 2
    const storage S1 = host::mul(&Y1, &ZZZ2);     // S1 = Y1*ZZZ2  < 2
    const storage S2 = host::mul(&Y2, &ZZZ1);     // S2 = Y2*ZZZ1  < 2
    const storage P = host::sub(&U2, &U1);        // P = U2-U1     < 2
    const storage R = host::sub(&S2, &S1);        // R = S2-S1     < 2

    const storage PP = host::sqr(&P);             // PP = P^2        < 2
    const storage PPP = host::mul(&P, &PP);       // PPP = P*PP      < 2
    const storage Q = host::mul(&U1, &PP);        // Q = U1*PP       < 2
    const storage t0 = host::sqr(&R);             // t0 = R^2        < 2
    const storage t1 = host::dbl(&Q);             // t1 = 2*Q        < 2
    const storage t2 = host::sub(&t0, &PPP);      // t2 = t0-PPP     < 2
    const storage X3 = host::sub(&t2, &t1);       // X3 = t2-t1      < 2
    const storage t3 = host::sub(&Q, &X3);        // t3 = Q-X3       < 2

    const storage t4 = host::mul(&S1, &PPP);      // t4 = S1*PPP     < 2
    const storage t5 = host::mul(&R, &t3);        // t5 = R*t3       < 2
    const storage Y3 = host::sub(&t5, &t4);       // Y3 = t5-t4      < 2

    const storage t6 = host::mul(&ZZ2, &PP);      // t6 = ZZ2*PP     < 2
    const storage ZZ3 = host::mul(&ZZ1, &t6);     // ZZ3 = ZZ1*t6    < 2
    const storage t7 = host::mul(&ZZZ2, &PPP);    // t7 = ZZZ2*PPP   < 2
    const storage ZZZ3 = host::mul(&ZZZ1, &t7);   // ZZZ3 = ZZZ1*t7  < 2

    return {X3, Y3, ZZ3, ZZZ3};
  }

static HOST_INLINE affine_point to_affine_point(const point_xyzz *p) {
    const storage zz_inv = host::inverse(p->zz);
    const storage zzz_inv = host::inverse(p->zzz);

    const storage x = host::mul(&p->x, &zz_inv);
    const storage y = host::mul(&p->y, &zzz_inv);

    return {x, y};
}

static HOST_INLINE affine_point from_montgomery(const affine_point *p) {
    const storage one = ONE;
    const storage x = host::mul(&p->x, &one);
    const storage y = host::mul(&p->y, &one);
    return {x, y};
}
}