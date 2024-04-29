#include "mont_op.cuh"

typedef struct {
    storage x;
    storage y;
} affine_point;

typedef struct {
    storage x;
    storage y;
    storage zz;
    storage zzz;
} point_xyzz;

// ----------------------------------------------------------------------------
// Coordinate conversions
// ----------------------------------------------------------------------------

static DEVICE_INLINE point_xyzz to_point_xyzz(const affine_point *p) {
    const storage X = p->x;
    const storage Y = p->y;
    return {X, Y, ONE_MONTGOMERY, ONE_MONTGOMERY};
}

static DEVICE_INLINE affine_point to_affine_point(const point_xyzz *p) {
    const storage zz_inv = inverse(p->zz);
    const storage zzz_inv = inverse(p->zzz);

    const storage x = mul(&p->x, &zz_inv);
    const storage y = mul(&p->y, &zzz_inv);

	return {x, y};
}

// ----------------------------------------------------------------------------
// Montgomery space conversions
// ----------------------------------------------------------------------------

static DEVICE_INLINE affine_point to_montgomery(const affine_point *p) {
    const storage rr = RR;
    const storage X = mul(&p->x, &rr);
    const storage Y = mul(&p->y, &rr);
    return {X, Y};
}

static DEVICE_INLINE affine_point from_montgomery(const affine_point *p) {
    const storage one = ONE;
    const storage x = mul(&p->x, &one);
    const storage y = mul(&p->y, &one);
    return {x, y};
}

// ----------------------------------------------------------------------------
// Various additions based on different coordinates and assumptions
// ----------------------------------------------------------------------------

// https://hyperelliptic.org/EFD/g1p/auto-shortw-xyzz.html#addition-add-2008-s
static DEVICE_INLINE point_xyzz madd_2008_s(const point_xyzz *p1, const affine_point *p2) {

    const storage rr = RR;
    
    const storage X1 = p1->x;     // < 2
    const storage Y1 = p1->y;     // < 2
    const storage ZZ1 = p1->zz;   // < 2
    const storage ZZZ1 = p1->zzz; // < 2
    const storage X2 = p2->x;     // < 2
    const storage Y2 = p2->y;     // < 2

    const storage U2 = mul(&X2, &ZZ1);  			// U2 = X2*ZZ1   < 2
    const storage U2_red = mul(&U2, &rr);           // transform back to Montgomery space
    const storage S2 = mul(&Y2, &ZZZ1); 			// S2 = Y2*ZZZ1  < 2
    const storage S2_red = mul(&S2, &rr);           // transform back to Montgomery space
    const storage P = sub(&U2_red, &X1);    		// P = U2-U1     < 2
    const storage R = sub(&S2_red, &Y1);    		// R = S2-S1     < 2
    const storage PP = sqr(&P);       				// PP = P^2        < 2
    const storage PPP = mul(&P, &PP);               // PPP = P*PP      < 2
    const storage Q = mul(&X1, &PP);                // Q = U1*PP       < 2
    const storage t0 = sqr(&R);       				// t0 = R^2        < 2
    const storage t1 = dbl(&Q);       				// t1 = 2*Q        < 2
    const storage t2 = sub(&t0, &PPP); 				// t2 = t0-PPP     < 2
    const storage X3 = sub(&t2, &t1);  				// X3 = t2-t1      < 2
    const storage t3 = sub(&Q, &X3);   				// t3 = Q-X3       < 2

    const storage t4 = mul(&Y1, &PPP);   			// t4 = Y1*PPP        < 2
    const storage t5 = mul(&R, &t3);     			// t5 = R*t3          < 2
    const storage Y3 = sub(&t5, &t4);    			// Y3 = t5-t4         < 2

    const storage ZZ3 = mul(&ZZ1, &PP);    			// ZZ3 = ZZ1*PP       < 2
    const storage ZZZ3 = mul(&ZZZ1, &PPP); 			// ZZZ3 = ZZZ1*PPP    < 2

    return {X3, Y3, ZZ3, ZZZ3};
}

// https://hyperelliptic.org/EFD/g1p/auto-shortw-xyzz.html#addition-add-2008-s
template <bool CHECK_ZERO = true, bool CHECK_DOUBLE = true>
static DEVICE_INLINE point_xyzz add_2008_s(const point_xyzz *p1, const point_xyzz *p2) {
    
    const storage X1 = p1->x;     // < 2
    const storage Y1 = p1->y;     // < 2
    const storage ZZ1 = p1->zz;   // < 2
    const storage ZZZ1 = p1->zzz; // < 2
    const storage X2 = p2->x;     // < 2
    const storage Y2 = p2->y;     // < 2
    const storage ZZ2 = p2->zz;   // < 2
    const storage ZZZ2 = p2->zzz; // < 2

    const storage U1 = mul(&X1, &ZZ2);      // U1 = X1*ZZ2   < 2
    const storage U2 = mul(&X2, &ZZ1);      // U2 = X2*ZZ1   < 2
    const storage S1 = mul(&Y1, &ZZZ2);     // S1 = Y1*ZZZ2  < 2
    const storage S2 = mul(&Y2, &ZZZ1);     // S2 = Y2*ZZZ1  < 2
    const storage P = sub(&U2, &U1);        // P = U2-U1     < 2
    const storage R = sub(&S2, &S1);        // R = S2-S1     < 2

    const storage PP = sqr(&P);             // PP = P^2        < 2
    const storage PPP = mul(&P, &PP);       // PPP = P*PP      < 2
    const storage Q = mul(&U1, &PP);        // Q = U1*PP       < 2
    const storage t0 = sqr(&R);             // t0 = R^2        < 2
    const storage t1 = dbl(&Q);             // t1 = 2*Q        < 2
    const storage t2 = sub(&t0, &PPP);      // t2 = t0-PPP     < 2
    const storage X3 = sub(&t2, &t1);       // X3 = t2-t1      < 2
    const storage t3 = sub(&Q, &X3);        // t3 = Q-X3       < 2

    const storage t4 = mul(&S1, &PPP);      // t4 = S1*PPP     < 2
    const storage t5 = mul(&R, &t3);        // t5 = R*t3       < 2
    const storage Y3 = sub(&t5, &t4);       // Y3 = t5-t4      < 2

    const storage t6 = mul(&ZZ2, &PP);      // t6 = ZZ2*PP     < 2
    const storage ZZ3 = mul(&ZZ1, &t6);     // ZZ3 = ZZ1*t6    < 2
    const storage t7 = mul(&ZZZ2, &PPP);    // t7 = ZZZ2*PPP   < 2
    const storage ZZZ3 = mul(&ZZZ1, &t7);   // ZZZ3 = ZZZ1*t7  < 2

    return {X3, Y3, ZZ3, ZZZ3};
  }