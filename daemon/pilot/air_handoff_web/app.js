var Dn = Object.defineProperty;
var Fn = (t, e, n) => e in t ? Dn(t, e, { enumerable: !0, configurable: !0, writable: !0, value: n }) : t[e] = n;
var g = (t, e, n) => Fn(t, typeof e != "symbol" ? e + "" : e, n);
/*! noble-ciphers - MIT License (c) 2023 Paul Miller (paulmillr.com) */
function se(t) {
  return t instanceof Uint8Array || ArrayBuffer.isView(t) && t.constructor.name === "Uint8Array" && "BYTES_PER_ELEMENT" in t && t.BYTES_PER_ELEMENT === 1;
}
const ce = (t) => t ? `"${t}" ` : "";
function kt(t, e = "") {
  if (typeof t != "number")
    throw new TypeError(ce(e) + "expected number, got " + typeof t);
  if (!Number.isSafeInteger(t) || t < 0)
    throw new RangeError(ce(e) + "expected integer >= 0, got " + t);
  return t;
}
function H(t, e, n = "") {
  if (se(t) && (e === void 0 || t.length === e))
    return t;
  e !== void 0 && kt(e, "length");
  const r = se(t), o = e !== void 0 ? ` of length ${e}` : "", i = r ? `length=${t.length}` : `type=${typeof t}`, s = ce(n) + "expected Uint8Array" + o + ", got " + i;
  throw r ? new RangeError(s) : new TypeError(s);
}
function Ue(t, e = !0) {
  if (t.destroyed)
    throw new Error("hash was destroyed");
  if (e && t.finished)
    throw new Error("digest() was already called");
}
function Mn(t, e) {
  H(t, void 0, "output");
  const n = e.outputLen;
  if (!(t.length >= n))
    throw new RangeError('"output" expected length >= ' + n);
}
function Zn(t, e) {
  if (Mn(t, e), !xt(t))
    throw new Error("invalid output, must be aligned");
}
function jn(t) {
  return new Uint8Array(t.buffer, t.byteOffset, t.byteLength);
}
function st(t) {
  return new Uint32Array(t.buffer, t.byteOffset, Math.floor(t.byteLength / 4));
}
function et(...t) {
  for (let e = 0; e < t.length; e++)
    t[e].fill(0);
}
function Yt(t) {
  return new DataView(t.buffer, t.byteOffset, t.byteLength);
}
const zt = new Uint8Array(new Uint32Array([287454020]).buffer)[0] === 68;
function on(t) {
  return t << 24 & 4278190080 | t << 8 & 16711680 | t >>> 8 & 65280 | t >>> 24 & 255;
}
const I = zt ? (t) => t : (t) => on(t) >>> 0;
function Pn(t) {
  for (let e = 0; e < t.length; e++)
    t[e] = on(t[e]);
  return t;
}
const pe = zt ? (t) => t : Pn;
function Yn(t, e) {
  if (t = H(t), e = H(e), t.length !== e.length)
    return !1;
  let n = 0;
  for (let r = 0; r < t.length; r++)
    n |= t[r] ^ e[r];
  return n === 0;
}
function zn(t, e, n) {
  const r = e, o = n || (() => []), i = (c, f) => r(f, ...o(c)).update(c).digest(), s = r(new Uint8Array(t), ...o(new Uint8Array(0)));
  return i.outputLen = s.outputLen, i.blockLen = s.blockLen, i.create = (c, ...f) => r(c, ...f), i;
}
const Xn = /* @__NO_SIDE_EFFECTS__ */ (t, e) => {
  function n(r, ...o) {
    if (H(r, void 0, "key"), t.nonceLength !== void 0) {
      const d = o[0];
      H(d, t.varSizeNonce ? void 0 : t.nonceLength, "nonce");
    }
    const i = t.tagLength, s = t.nonceLength !== void 0 ? 1 : 0;
    if (!t.withAAD) {
      for (let d = s; d < o.length; d++)
        if (se(o[d]))
          throw new Error("AAD not supported");
    }
    t.withAAD && o[s] !== void 0 && H(o[s], void 0, "AAD");
    const c = e(r, ...o), f = (d, l) => {
      if (l !== void 0) {
        if (d !== 2)
          throw new Error("cipher output not supported");
        H(l, void 0, "output");
      }
    };
    let a = !1;
    return {
      encrypt(d, l) {
        if (a)
          throw new Error("cannot encrypt() twice with same key + nonce");
        return a = !0, H(d, void 0, "data"), f(c.encrypt.length, l), c.encrypt(d, l);
      },
      decrypt(d, l) {
        if (H(d, void 0, "data"), i && d.length < i)
          throw new Error('"ciphertext" expected length >= tagLength=' + i);
        return f(c.decrypt.length, l), c.decrypt(d, l);
      }
    };
  }
  return Object.assign(n, t), n;
};
function Wn(t, e, n = !0) {
  if (e === void 0)
    return new Uint8Array(t);
  if (H(e, t, "output"), n && !xt(e))
    throw new Error("invalid output, must be aligned");
  return e;
}
function Kn(t, e, n) {
  kt(t), kt(e);
  const r = new Uint8Array(16), o = Yt(r);
  return o.setBigUint64(0, BigInt(e), n), o.setBigUint64(8, BigInt(t), n), r;
}
function xt(t) {
  return t.byteOffset % 4 === 0;
}
function Bt(t) {
  return Uint8Array.from(H(t));
}
function rn(t = 32) {
  kt(t, "bytesLength");
  const e = typeof globalThis == "object" ? globalThis.crypto : null;
  if (typeof (e == null ? void 0 : e.getRandomValues) != "function")
    throw new Error("crypto.getRandomValues must be defined");
  if (t > 65536)
    throw new RangeError(`"bytesLength" expected <= 65536, got ${t}`);
  return e.getRandomValues(new Uint8Array(t));
}
const nt = 16, sn = /* @__PURE__ */ new Uint8Array(16), pt = /* @__PURE__ */ st(sn), Vn = 225, Gn = (t, e, n, r) => {
  const o = r & 1;
  return {
    s3: n << 31 | r >>> 1,
    s2: e << 31 | n >>> 1,
    s1: t << 31 | e >>> 1,
    // NIST SP 800-38D §6.3 applies `V >> 1` and XORs R on carry. In this
    // 4x32-bit split, R = 0xe1 || 0^120 lives in the top byte of s0.
    s0: t >>> 1 ^ Vn << 24 & -(o & 1)
    // reduce % poly
  };
}, St = (t) => (t >>> 0 & 255) << 24 | (t >>> 8 & 255) << 16 | (t >>> 16 & 255) << 8 | t >>> 24 & 255 | 0, Jn = (t) => t > 64 * 1024 ? 8 : t > 1024 ? 4 : 2;
class Qn {
  // We select bits per window adaptively based on expectedLength
  constructor(e, n) {
    g(this, "blockLen", nt);
    g(this, "outputLen", nt);
    g(this, "s0", 0);
    g(this, "s1", 0);
    g(this, "s2", 0);
    g(this, "s3", 0);
    g(this, "finished", !1);
    g(this, "destroyed", !1);
    g(this, "t");
    g(this, "W");
    g(this, "windowSize");
    H(e, 16, "key"), e = Bt(e);
    const r = Yt(e);
    let o = r.getUint32(0, !1), i = r.getUint32(4, !1), s = r.getUint32(8, !1), c = r.getUint32(12, !1);
    const f = [];
    for (let w = 0; w < 128; w++)
      f.push({ s0: St(o), s1: St(i), s2: St(s), s3: St(c) }), { s0: o, s1: i, s2: s, s3: c } = Gn(o, i, s, c);
    const a = Jn(n || 1024);
    if (![1, 2, 4, 8].includes(a))
      throw new Error("ghash: invalid window size, expected 2, 4 or 8");
    this.W = a;
    const d = 128 / a, l = this.windowSize = 2 ** a, h = [];
    for (let w = 0; w < d; w++)
      for (let m = 0; m < l; m++) {
        let x = 0, v = 0, y = 0, R = 0;
        for (let p = 0; p < a; p++) {
          if (!(m >>> a - p - 1 & 1))
            continue;
          const { s0: b, s1: A, s2: S, s3: L } = f[a * w + p];
          x ^= b, v ^= A, y ^= S, R ^= L;
        }
        h.push({ s0: x, s1: v, s2: y, s3: R });
      }
    this.t = h;
  }
  _updateBlock(e, n, r, o) {
    e ^= this.s0, n ^= this.s1, r ^= this.s2, o ^= this.s3;
    const { W: i, t: s, windowSize: c } = this;
    let f = 0, a = 0, u = 0, d = 0;
    const l = (1 << i) - 1;
    let h = 0;
    for (const w of [e, n, r, o])
      for (let m = 0; m < 4; m++) {
        const x = w >>> 8 * m & 255;
        for (let v = 8 / i - 1; v >= 0; v--) {
          const y = x >>> i * v & l, { s0: R, s1: p, s2: E, s3: b } = s[h * c + y];
          f ^= R, a ^= p, u ^= E, d ^= b, h += 1;
        }
      }
    this.s0 = f, this.s1 = a, this.s2 = u, this.s3 = d;
  }
  update(e) {
    Ue(this), H(e), e = Bt(e);
    const n = st(e), r = Math.floor(e.length / nt), o = e.length % nt;
    for (let i = 0; i < r; i++)
      this._updateBlock(I(n[i * 4 + 0]), I(n[i * 4 + 1]), I(n[i * 4 + 2]), I(n[i * 4 + 3]));
    return o && (sn.set(e.subarray(r * nt)), this._updateBlock(I(pt[0]), I(pt[1]), I(pt[2]), I(pt[3])), et(pt)), this;
  }
  destroy() {
    this.destroyed = !0;
    const { t: e } = this;
    for (const n of e)
      n.s0 = 0, n.s1 = 0, n.s2 = 0, n.s3 = 0;
  }
  digestInto(e) {
    Ue(this), Zn(e, this), this.finished = !0;
    const { s0: n, s1: r, s2: o, s3: i } = this, s = st(e);
    s[0] = n, s[1] = r, s[2] = o, s[3] = i, zt || pe(s.subarray(0, nt / 4));
  }
  digest() {
    const e = new Uint8Array(nt);
    return this.digestInto(e), this.destroy(), e;
  }
}
const $e = /* @__PURE__ */ zn(16, (t, e) => new Qn(t, e), (t) => [t.length]), fe = 16, to = 4, It = /* @__PURE__ */ new Uint8Array(fe), eo = 283;
function no(t) {
  if (![16, 24, 32].includes(t.length))
    throw new Error('"aes key" expected Uint8Array of length 16/24/32, got length=' + t.length);
}
function ge(t) {
  return t << 1 ^ eo & -(t >> 7);
}
function Ne(t, e) {
  let n = 0;
  for (; e > 0; e >>= 1)
    n ^= t & -(e & 1), t = ge(t);
  return n;
}
const oo = /* @__PURE__ */ (() => {
  const t = new Uint8Array(256);
  for (let n = 0, r = 1; n < 256; n++, r ^= ge(r))
    t[n] = r;
  const e = new Uint8Array(256);
  e[0] = 99;
  for (let n = 0; n < 255; n++) {
    let r = t[255 - n];
    r |= r << 8, e[t[n]] = (r ^ r >> 4 ^ r >> 5 ^ r >> 6 ^ r >> 7 ^ 99) & 255;
  }
  return et(t), e;
})(), ro = (t) => t << 24 | t >>> 8, ne = (t) => t << 8 | t >>> 24;
function io(t, e) {
  if (t.length !== 256)
    throw new Error("wrong sbox length");
  const n = new Uint32Array(256).map((a, u) => e(t[u])), r = n.map(ne), o = r.map(ne), i = o.map(ne), s = new Uint32Array(256 * 256), c = new Uint32Array(256 * 256), f = new Uint16Array(256 * 256);
  for (let a = 0; a < 256; a++)
    for (let u = 0; u < 256; u++) {
      const d = a * 256 + u;
      s[d] = n[a] ^ r[u], c[d] = o[a] ^ i[u], f[d] = t[a] << 8 | t[u];
    }
  return { sbox: t, sbox2: f, T0: n, T1: r, T2: o, T3: i, T01: s, T23: c };
}
const cn = /* @__PURE__ */ io(oo, (t) => Ne(t, 3) << 24 | t << 16 | t << 8 | Ne(t, 2)), so = /* @__PURE__ */ (() => {
  const t = new Uint8Array(16);
  for (let e = 0, n = 1; e < 16; e++, n = ge(n))
    t[e] = n;
  return t;
})();
function co(t) {
  H(t);
  const e = t.length;
  no(t);
  const { sbox2: n } = cn, r = [];
  (!zt || !xt(t)) && r.push(t = Bt(t));
  const o = pe(st(t)), i = o.length, s = (f) => wt(n, f, f, f, f), c = new Uint32Array(e + 28);
  c.set(o);
  for (let f = i; f < c.length; f++) {
    let a = c[f - 1];
    f % i === 0 ? a = s(ro(a)) ^ so[f / i - 1] : i > 6 && f % i === 4 && (a = s(a)), c[f] = c[f - i] ^ a;
  }
  return et(...r), c;
}
function Lt(t, e, n, r, o, i) {
  return t[n << 8 & 65280 | r >>> 8 & 255] ^ e[o >>> 8 & 65280 | i >>> 24 & 255];
}
function wt(t, e, n, r, o) {
  return t[e & 255 | n & 65280] | t[r >>> 16 & 255 | o >>> 16 & 65280] << 16;
}
function qe(t, e, n, r, o) {
  const { sbox2: i, T01: s, T23: c } = cn;
  let f = 0;
  e ^= t[f++], n ^= t[f++], r ^= t[f++], o ^= t[f++];
  const a = t.length / 4 - 2;
  for (let w = 0; w < a; w++) {
    const m = t[f++] ^ Lt(s, c, e, n, r, o), x = t[f++] ^ Lt(s, c, n, r, o, e), v = t[f++] ^ Lt(s, c, r, o, e, n), y = t[f++] ^ Lt(s, c, o, e, n, r);
    e = m, n = x, r = v, o = y;
  }
  const u = t[f++] ^ wt(i, e, n, r, o), d = t[f++] ^ wt(i, n, r, o, e), l = t[f++] ^ wt(i, r, o, e, n), h = t[f++] ^ wt(i, o, e, n, r);
  return { s0: u, s1: d, s2: l, s3: h };
}
function Ct(t, e, n, r, o) {
  H(n, fe, "nonce"), H(r), o = Wn(r.length, o);
  const i = n, s = st(i), c = Yt(i), f = st(r), a = st(o), u = e ? 0 : 12, d = r.length;
  let l = c.getUint32(u, e);
  for (let w = 0; w + 4 <= f.length; w += 4) {
    const { s0: m, s1: x, s2: v, s3: y } = qe(t, I(s[0]), I(s[1]), I(s[2]), I(s[3]));
    a[w + 0] = f[w + 0] ^ I(m), a[w + 1] = f[w + 1] ^ I(x), a[w + 2] = f[w + 2] ^ I(v), a[w + 3] = f[w + 3] ^ I(y), l = l + 1 >>> 0, c.setUint32(u, l, e);
  }
  const h = fe * Math.floor(f.length / to);
  if (h < d) {
    const { s0: w, s1: m, s2: x, s3: v } = qe(t, I(s[0]), I(s[1]), I(s[2]), I(s[3])), y = new Uint32Array([w, m, x, v]);
    pe(y);
    const R = jn(y);
    for (let p = h, E = 0; p < d; p++, E++)
      o[p] = r[p] ^ R[E];
    et(y);
  }
  return o;
}
function fo(t, e, n, r, o) {
  const i = o ? o.length : 0, s = t.create(n, r.length + i);
  o && s.update(o);
  const c = Kn(8 * r.length, 8 * i, e);
  s.update(r), s.update(c);
  const f = s.digest();
  return et(c), f;
}
const Xt = /* @__PURE__ */ Xn({ blockSize: 16, nonceLength: 12, tagLength: 16, withAAD: !0, varSizeNonce: !0 }, function(e, n, r) {
  if (n.length < 8)
    throw new Error("aes/gcm: invalid nonce length");
  const o = 16;
  function i(c, f, a) {
    const u = fo($e, !1, c, a, r);
    for (let d = 0; d < f.length; d++)
      u[d] ^= f[d];
    return u;
  }
  function s() {
    const c = co(e), f = It.slice(), a = It.slice();
    if (Ct(c, !1, a, a, f), n.length === 12)
      a.set(n);
    else {
      const d = It.slice();
      Yt(d).setBigUint64(8, BigInt(n.length * 8), !1);
      const h = $e.create(f).update(n).update(d);
      h.digestInto(a), h.destroy();
    }
    const u = Ct(c, !1, a, It);
    return { xk: c, authKey: f, counter: a, tagMask: u };
  }
  return {
    encrypt(c) {
      const { xk: f, authKey: a, counter: u, tagMask: d } = s(), l = new Uint8Array(c.length + o), h = [f, a, u, d];
      xt(c) || h.push(c = Bt(c)), Ct(f, !1, u, c, l.subarray(0, c.length));
      const w = i(a, d, l.subarray(0, l.length - o));
      return h.push(w), l.set(w, c.length), et(...h), l;
    },
    decrypt(c) {
      const { xk: f, authKey: a, counter: u, tagMask: d } = s(), l = [f, a, d, u];
      xt(c) || l.push(c = Bt(c));
      const h = c.subarray(0, -o), w = c.subarray(-o), m = i(a, d, h);
      if (l.push(m), !Yn(m, w))
        throw et(...l), new Error("aes-gcm: invalid tag");
      const x = Ct(f, !1, u, h);
      return et(...l), x;
    }
  };
}), ao = (t) => t / 2 ** 32 | 0, uo = (t) => t >>> 0;
function lo(t, e, n, r) {
  const o = ao(n), i = uo(n);
  t.setUint32(e, r ? i : o, r), t.setUint32(e + 4, r ? o : i, r);
}
function ae(t) {
  return t instanceof Uint8Array || ArrayBuffer.isView(t) && t.constructor.name === "Uint8Array" && "BYTES_PER_ELEMENT" in t && t.BYTES_PER_ELEMENT === 1;
}
const ue = (t) => t ? `"${t}" ` : "";
function ft(t, e = "") {
  if (typeof t != "number")
    throw new TypeError(ue(e) + "expected number, got " + typeof t);
  if (!Number.isSafeInteger(t) || t < 0)
    throw new RangeError(ue(e) + "expected integer >= 0, got " + t);
  return t;
}
function Q(t, e, n = "") {
  if (ae(t) && (e === void 0 || t.length === e))
    return t;
  e !== void 0 && ft(e, "length");
  const r = ae(t), o = e !== void 0 ? ` of length ${e}` : "", i = r ? `length=${t.length}` : `type=${typeof t}`, s = ue(n) + "expected Uint8Array" + o + ", got " + i;
  throw r ? new RangeError(s) : new TypeError(s);
}
function we(t) {
  if (typeof t != "function" || typeof t.create != "function")
    throw new TypeError("expected hash wrapped by utils.createHasher");
  if (ft(t.outputLen), ft(t.blockLen), t.outputLen < 1 || t.blockLen < 1)
    throw new Error("hash blockLen / outputLen must be >= 1");
}
const He = (t, e) => {
  if (t === null || typeof t != "object" || Array.isArray(t))
    throw new TypeError((e === "object" ? "" : `"${e}" `) + "expected object, got type=" + typeof t);
};
function Dt(t, e = !0) {
  if (t.destroyed)
    throw new Error("hash was destroyed");
  if (e && t.finished)
    throw new Error("digest() was already called");
}
function fn(t, e) {
  Q(t, void 0, "output");
  const n = e.outputLen;
  if (!(t.length >= n))
    throw new RangeError('"output" expected length >= ' + n);
}
function rt(...t) {
  for (let e = 0; e < t.length; e++)
    t[e].fill(0);
}
function oe(t) {
  return new DataView(t.buffer, t.byteOffset, t.byteLength);
}
function K(t, e) {
  return t << 32 - e | t >>> e;
}
const an = /* @ts-ignore */ typeof Uint8Array.from([]).toHex == "function" && typeof Uint8Array.fromHex == "function", ho = /* @__PURE__ */ Array.from({ length: 256 }, (t, e) => e.toString(16).padStart(2, "0"));
function Wt(t) {
  if (Q(t), an)
    return t.toHex();
  let e = "";
  for (let n = 0; n < t.length; n++)
    e += ho[t[n]];
  return e;
}
function ke(t) {
  return t >= 48 && t <= 57 ? t - 48 : t >= 65 && t <= 70 ? t - 55 : t >= 97 && t <= 102 ? t - 87 : void 0;
}
function un(t) {
  if (typeof t != "string")
    throw new TypeError("hex string expected, got " + typeof t);
  if (an)
    try {
      return Uint8Array.fromHex(t);
    } catch (o) {
      throw o instanceof SyntaxError ? new RangeError(o.message) : o;
    }
  const e = t.length, n = e / 2;
  if (e % 2)
    throw new RangeError("hex string expected, got unpadded hex of length " + e);
  const r = new Uint8Array(n);
  for (let o = 0, i = 0; o < n; o++, i += 2) {
    const s = ke(t.charCodeAt(i)), c = ke(t.charCodeAt(i + 1));
    if (s === void 0 || c === void 0) {
      const f = t[i] + t[i + 1];
      throw new RangeError('hex string expected, got non-hex character "' + f + '" at index ' + i);
    }
    r[o] = s * 16 + c;
  }
  return r;
}
function De(...t) {
  let e = 0;
  for (let r = 0; r < t.length; r++) {
    const o = t[r];
    Q(o), e += o.length;
  }
  const n = new Uint8Array(e);
  for (let r = 0, o = 0; r < t.length; r++) {
    const i = t[r];
    n.set(i, o), o += i.length;
  }
  return n;
}
function po(t, e, n = "opts") {
  return He(t, "defaults"), e !== void 0 && He(e, n), Object.assign(t, e);
}
function go(t, e = {}) {
  if (typeof t != "function")
    throw new TypeError('"hashCons" expected function, got type=' + typeof t);
  e = po({}, e, "info");
  const n = (o, i) => t(i).update(o).digest(), r = t(void 0);
  return n.outputLen = r.outputLen, n.blockLen = r.blockLen, n.canXOF = r.canXOF, n.create = (o) => t(o), Object.assign(n, e), Object.freeze(n);
}
function wo(t = 32) {
  ft(t, "bytesLength");
  const e = typeof globalThis == "object" ? globalThis.crypto : null;
  if (typeof (e == null ? void 0 : e.getRandomValues) != "function")
    throw new Error("crypto.getRandomValues must be defined");
  if (t > 65536)
    throw new RangeError(`"bytesLength" expected <= 65536, got ${t}`);
  return e.getRandomValues(new Uint8Array(t));
}
const yo = (t) => ({
  // Current NIST hashAlgs suffixes used here fit in one DER subidentifier octet.
  // Larger suffix values would need base-128 OID encoding and a different length byte.
  oid: Uint8Array.from([6, 9, 96, 134, 72, 1, 101, 3, 4, 2, t])
});
function mo(t, e, n) {
  return t & e ^ ~t & n;
}
function bo(t, e, n) {
  return t & e ^ t & n ^ e & n;
}
class Eo {
  constructor(e, n, r, o) {
    g(this, "blockLen");
    g(this, "outputLen");
    g(this, "canXOF", !1);
    g(this, "padOffset");
    g(this, "isLE");
    // For partial updates less than block size
    g(this, "buffer");
    g(this, "view");
    g(this, "finished", !1);
    g(this, "length", 0);
    g(this, "pos", 0);
    g(this, "destroyed", !1);
    this.blockLen = e, this.outputLen = n, this.padOffset = r, this.isLE = o, this.buffer = new Uint8Array(e), this.view = oe(this.buffer);
  }
  update(e) {
    Dt(this), Q(e);
    const { view: n, buffer: r, blockLen: o } = this, i = e.length;
    let s = !1;
    for (let c = 0; c < i; ) {
      const f = Math.min(o - this.pos, i - c);
      if (f === o) {
        const a = oe(e);
        for (; o <= i - c; c += o)
          this.process(a, c);
        s = !0;
        continue;
      }
      r.set(c === 0 && f === i ? e : e.subarray(c, c + f), this.pos), this.pos += f, c += f, this.pos === o && (this.process(n, 0), this.pos = 0, s = !0);
    }
    return this.length += e.length, s && this.roundClean(), this;
  }
  digestInto(e) {
    Dt(this), fn(e, this), this.finished = !0;
    const { buffer: n, view: r, blockLen: o, isLE: i } = this;
    let { pos: s } = this;
    n[s++] = 128, n.fill(0, s), this.padOffset > o - s && (this.process(r, 0), n.fill(0)), lo(r, o - 8, this.length * 8, i), this.process(r, 0), this.roundClean();
    const c = e === n ? r : oe(e), f = this.outputLen, a = f / 4, u = this.get();
    if (f % 4 || a > u.length)
      throw new Error("invalid outputLen");
    for (let d = 0; d < a; d++)
      c.setUint32(4 * d, u[d], i);
  }
  digest() {
    const { buffer: e, outputLen: n } = this;
    this.digestInto(e);
    const r = e.slice(0, n);
    return this.destroy(), r;
  }
  _cloneIntoMeta(e) {
    const { buffer: n, length: r, finished: o, destroyed: i, pos: s } = this;
    return e.destroyed = i, e.finished = o, e.length = r, e.pos = s, s && e.buffer.set(n), e;
  }
  clone() {
    return this._cloneInto();
  }
}
const xo = /* @__PURE__ */ Uint32Array.from([
  1779033703,
  3144134277,
  1013904242,
  2773480762,
  1359893119,
  2600822924,
  528734635,
  1541459225
]), Bo = /* @__PURE__ */ Uint32Array.from([
  1116352408,
  1899447441,
  3049323471,
  3921009573,
  961987163,
  1508970993,
  2453635748,
  2870763221,
  3624381080,
  310598401,
  607225278,
  1426881987,
  1925078388,
  2162078206,
  2614888103,
  3248222580,
  3835390401,
  4022224774,
  264347078,
  604807628,
  770255983,
  1249150122,
  1555081692,
  1996064986,
  2554220882,
  2821834349,
  2952996808,
  3210313671,
  3336571891,
  3584528711,
  113926993,
  338241895,
  666307205,
  773529912,
  1294757372,
  1396182291,
  1695183700,
  1986661051,
  2177026350,
  2456956037,
  2730485921,
  2820302411,
  3259730800,
  3345764771,
  3516065817,
  3600352804,
  4094571909,
  275423344,
  430227734,
  506948616,
  659060556,
  883997877,
  958139571,
  1322822218,
  1537002063,
  1747873779,
  1955562222,
  2024104815,
  2227730452,
  2361852424,
  2428436474,
  2756734187,
  3204031479,
  3329325298
]), tt = /* @__PURE__ */ new Uint32Array(64);
class vo extends Eo {
  constructor(n, r) {
    super(64, n, 8, !1);
    // We cannot use array here since array allows indexing by variable
    // which means optimizer/compiler cannot use registers.
    // Numeric initializers matter: starting the fields as `undefined` changes
    // V8's field representation and makes sha256 3x slower (measured).
    g(this, "A", 0);
    g(this, "B", 0);
    g(this, "C", 0);
    g(this, "D", 0);
    g(this, "E", 0);
    g(this, "F", 0);
    g(this, "G", 0);
    g(this, "H", 0);
    this.A = r[0] | 0, this.B = r[1] | 0, this.C = r[2] | 0, this.D = r[3] | 0, this.E = r[4] | 0, this.F = r[5] | 0, this.G = r[6] | 0, this.H = r[7] | 0;
  }
  get() {
    const { A: n, B: r, C: o, D: i, E: s, F: c, G: f, H: a } = this;
    return [n, r, o, i, s, c, f, a];
  }
  // prettier-ignore
  set(n, r, o, i, s, c, f, a) {
    this.A = n | 0, this.B = r | 0, this.C = o | 0, this.D = i | 0, this.E = s | 0, this.F = c | 0, this.G = f | 0, this.H = a | 0;
  }
  _cloneInto(n) {
    return (n || (n = new this.constructor())).set(...this.get()), this._cloneIntoMeta(n);
  }
  process(n, r) {
    for (let l = 0; l < 16; l++, r += 4)
      tt[l] = n.getUint32(r, !1);
    for (let l = 16; l < 64; l++) {
      const h = tt[l - 15], w = tt[l - 2], m = K(h, 7) ^ K(h, 18) ^ h >>> 3, x = K(w, 17) ^ K(w, 19) ^ w >>> 10;
      tt[l] = x + tt[l - 7] + m + tt[l - 16] | 0;
    }
    let { A: o, B: i, C: s, D: c, E: f, F: a, G: u, H: d } = this;
    for (let l = 0; l < 64; l++) {
      const h = K(f, 6) ^ K(f, 11) ^ K(f, 25), w = d + h + mo(f, a, u) + Bo[l] + tt[l] | 0, x = (K(o, 2) ^ K(o, 13) ^ K(o, 22)) + bo(o, i, s) | 0;
      d = u, u = a, a = f, f = c + w | 0, c = s, s = i, i = o, o = w + x | 0;
    }
    o = o + this.A | 0, i = i + this.B | 0, s = s + this.C | 0, c = c + this.D | 0, f = f + this.E | 0, a = a + this.F | 0, u = u + this.G | 0, d = d + this.H | 0, this.set(o, i, s, c, f, a, u, d);
  }
  roundClean() {
    rt(tt);
  }
  destroy() {
    this.destroyed = !0, this.set(0, 0, 0, 0, 0, 0, 0, 0), rt(this.buffer);
  }
}
class Ro extends vo {
  constructor() {
    super(32, xo);
  }
}
const mt = /* @__PURE__ */ go(
  () => new Ro(),
  /* @__PURE__ */ yo(1)
);
/*! noble-curves - MIT License (c) 2022 Paul Miller (paulmillr.com) */
function dn(t, e, n = () => {
}) {
  if (!Array.isArray(t))
    throw new TypeError(`"${e}" expected array, got type=${typeof t}`);
  for (let r = 0; r < t.length; r++)
    n(t[r], `${e}[${r}]`);
  return t;
}
const at = (t, e, n) => Q(t, e, n), ln = ft;
function ut(t, e = "object") {
  if (t === null || typeof t != "object" || Array.isArray(t))
    throw new TypeError(e === "object" ? "expected valid options object" : `"${e}" expected object, got type=${typeof t}`);
  return t;
}
function bt(t, e) {
  if (typeof t != "function")
    throw new TypeError(`"${e}" is invalid: expected function, got ${typeof t}`);
  return t;
}
const Oo = Wt, To = (t) => un(t), hn = ae, pn = (t) => wo(t), Ft = /* @__PURE__ */ BigInt(0), Fe = /* @__PURE__ */ BigInt(1), _o = (t) => t ? `"${t}" ` : "";
function Kt(t, e = "") {
  if (typeof t != "boolean")
    throw new TypeError(_o(e) + "expected boolean, got type=" + typeof t);
  return t;
}
function Ao(t) {
  if (typeof t == "bigint") {
    if (!ct(t))
      throw new RangeError("positive bigint expected, got " + t);
  } else
    ln(t);
  return t;
}
function de(t, e = "") {
  if (typeof t != "number") {
    const n = e && `"${e}" `;
    throw new TypeError(n + "expected number, got type=" + typeof t);
  }
  if (!Number.isSafeInteger(t)) {
    const n = e && `"${e}" `;
    throw new RangeError(n + "expected safe integer, got " + t);
  }
}
function gn(t) {
  if (typeof t != "string")
    throw new TypeError("hex string expected, got " + typeof t);
  return t === "" ? Ft : BigInt("0x" + t);
}
function wn(t) {
  return gn(Wt(t));
}
function Mt(t) {
  return gn(Wt(vt(Q(t)).reverse()));
}
function yn(t, e) {
  if (ft(e), e === 0)
    throw new Error("zero output length is invalid");
  t = Ao(t);
  const n = e * 2, r = t.toString(16);
  if (r.length > n)
    throw new RangeError("number is too large");
  return un(r.padStart(n, "0"));
}
function mn(t, e) {
  return yn(t, e).reverse();
}
function vt(t) {
  return Uint8Array.from(at(t));
}
function ct(t) {
  return typeof t == "bigint" && Ft <= t;
}
function bn(t, e, n) {
  return ct(t) && ct(e) && ct(n) && e <= t && t < n;
}
function Et(t, e, n, r) {
  if (!bn(e, n, r))
    throw new RangeError("expected valid " + t + ": " + n + " <= n < " + r + ", got " + e);
}
function So(t) {
  if (t < Ft)
    throw new Error("expected non-negative bigint, got " + t);
  return t === Ft ? 0 : t.toString(2).length;
}
const Io = (t) => (de(t, "n"), (Fe << BigInt(t)) - Fe);
function Zt(t, e = {}, n = {}, r = "object") {
  ut(t, r), ut(e, "fields"), ut(n, "optFields");
  function o(s, c, f) {
    const a = r === "object" ? `param "${String(s)}"` : `"${r}.${String(s)}"`, u = t[s];
    if (!Object.hasOwn(t, s) && (f ? u !== void 0 : c !== "function"))
      throw new TypeError(`${a} is invalid: expected own property`);
    if (f && u === void 0)
      return;
    const d = typeof u;
    if (d !== c || u === null)
      throw new TypeError(`${a} is invalid: expected ${c}, got ${d}`);
  }
  const i = (s, c) => Object.entries(s).forEach(([f, a]) => o(f, a, c));
  i(e, !1), i(n, !0);
}
/*! noble-curves - MIT License (c) 2022 Paul Miller (paulmillr.com) */
const N = /* @__PURE__ */ BigInt(0), T = /* @__PURE__ */ BigInt(1), it = /* @__PURE__ */ BigInt(2), En = /* @__PURE__ */ BigInt(3), ye = /* @__PURE__ */ BigInt(4), xn = /* @__PURE__ */ BigInt(5), Lo = /* @__PURE__ */ BigInt(7), Bn = /* @__PURE__ */ BigInt(8), Co = /* @__PURE__ */ BigInt(9), Uo = /* @__PURE__ */ BigInt(15), vn = /* @__PURE__ */ BigInt(16), $o = /* @__PURE__ */ BigInt("0x10000000000000000");
function O(t, e) {
  if (e <= N)
    throw new Error("mod: expected positive modulus, got " + e);
  const n = t % e;
  return n >= N ? n : e + n;
}
function No(t, e, n) {
  if (n <= T)
    throw new Error("pow: expected modulus > 1, got " + n);
  if (typeof e != "bigint")
    throw new TypeError("invalid exponent: expected bigint, got " + typeof e);
  if (e < N)
    throw new Error("invalid exponent, negatives unsupported");
  if (e === N)
    return T;
  if (e === T)
    return t;
  let r = t % n;
  if (r < N && (r += n), e < $o) {
    let c = T;
    for (; e > N; )
      e & T && (c = c * r % n), r = r * r % n, e >>= T;
    return c;
  }
  const o = [];
  for (; e > N; )
    o.push(Number(e & Uo)), e >>= ye;
  const i = new Array(16);
  i[0] = T, i[1] = r;
  for (let c = 2; c < 16; c++)
    i[c] = i[c - 1] * r % n;
  let s = i[o[o.length - 1]];
  for (let c = o.length - 2; c >= 0; c--) {
    s = s * s % n, s = s * s % n, s = s * s % n, s = s * s % n;
    const f = o[c];
    f !== 0 && (s = s * i[f] % n);
  }
  return s;
}
function X(t, e, n) {
  if (n <= T)
    throw new Error("pow2: expected modulus > 1, got " + n);
  if (e < N)
    throw new Error("pow2: expected non-negative exponent, got " + e);
  let r = t;
  for (; e-- > N; )
    r *= r, r %= n;
  return r;
}
function Me(t, e) {
  if (t === N)
    throw new Error("invert: expected non-zero number");
  if (e <= T)
    throw new Error("invert: expected modulus > 1, got " + e);
  let n = O(t, e), r = e, o = N, i = T;
  for (; n !== N; ) {
    const c = r / n, f = r - n * c, a = o - i * c;
    r = n, n = f, o = i, i = a;
  }
  if (r !== T)
    throw new Error("invert: does not exist");
  return O(o, e);
}
function me(t, e, n) {
  const r = t;
  if (!r.eql(r.sqr(e), n))
    throw new Error("Cannot find square root");
}
function be(t, e) {
  if ((t & T) === N)
    throw new Error(e + ": expected odd modulus, got " + t);
}
function Rn(t, e) {
  const n = t, r = (n.ORDER + T) / ye, o = n.pow(e, r);
  return me(n, o, e), o;
}
function qo(t, e) {
  const n = t, r = (n.ORDER - xn) / Bn, o = n.mul(e, it), i = n.pow(o, r), s = n.mul(e, i), c = n.mul(n.mul(s, it), i), f = n.mul(s, n.sub(c, n.ONE));
  return me(n, f, e), f;
}
function Ho(t) {
  const e = Ee(t), n = On(t), r = n(e, e.neg(e.ONE)), o = n(e, r), i = n(e, e.neg(r)), s = (t + Lo) / vn;
  return ((c, f) => {
    const a = c;
    let u = a.pow(f, s), d = a.mul(u, r);
    const l = a.mul(u, o), h = a.mul(u, i), w = a.eql(a.sqr(d), f), m = a.eql(a.sqr(l), f);
    u = a.cmov(u, d, w), d = a.cmov(h, l, m);
    const x = a.eql(a.sqr(d), f), v = a.cmov(u, d, x);
    return me(a, v, f), v;
  });
}
function On(t) {
  if (t < En)
    throw new Error("sqrt is not defined for small field");
  be(t, "tonelliShanks");
  let e = t - T, n = 0;
  for (; e % it === N; )
    e /= it, n++;
  let r = it;
  const o = Ee(t);
  for (; jt(o, r) === 1; )
    if (r++ > 1e3)
      throw new Error("Cannot find square root: probably non-prime P");
  if (n === 1)
    return Rn;
  let i = o.pow(r, e);
  const s = (e + T) / it;
  return function(f, a) {
    const u = f;
    if (u.is0(a))
      return a;
    if (jt(u, a) !== 1)
      throw new Error("Cannot find square root");
    let d = n, l = u.mul(u.ONE, i), h = u.pow(a, e), w = u.pow(a, s);
    for (; !u.eql(h, u.ONE); ) {
      if (u.is0(h))
        throw new Error("Cannot find square root: probably non-prime P");
      let m = 1, x = u.sqr(h);
      for (; !u.eql(x, u.ONE); )
        if (m++, x = u.sqr(x), m === d)
          throw new Error("Cannot find square root");
      const v = T << BigInt(d - m - 1), y = u.pow(l, v);
      d = m, l = u.sqr(y), h = u.mul(h, l), w = u.mul(w, y);
    }
    return w;
  };
}
function ko(t) {
  return be(t, "Fp.sqrt"), t % ye === En ? Rn : t % Bn === xn ? qo : t % vn === Co ? Ho(t) : On(t);
}
const Do = (t, e) => (O(t, e) & T) === T, Fo = [
  "create",
  "isValid",
  "is0",
  "neg",
  "inv",
  "sqrt",
  "sqr",
  "eql",
  "add",
  "sub",
  "mul",
  "pow",
  "div",
  "addN",
  "subN",
  "mulN",
  "sqrN"
];
function Rt(t) {
  if (ut(t, "field"), typeof t.ORDER != "bigint")
    throw new TypeError('param "ORDER" is invalid: expected bigint, got ' + typeof t.ORDER);
  de(t.BYTES, "BYTES"), de(t.BITS, "BITS");
  for (const e of Fo)
    bt(t[e], "field." + e);
  if (t.BYTES < 1 || t.BITS < 1)
    throw new Error("invalid field: expected BYTES/BITS > 0");
  if (t.ORDER <= T)
    throw new Error("invalid field: expected ORDER > 1, got " + t.ORDER);
  return t;
}
function Tn(t, e, n = !1) {
  Rt(t), dn(e, "nums"), Kt(n, "passZero");
  const r = t, o = new Array(e.length).fill(n ? r.ZERO : void 0), i = e.reduce((c, f, a) => r.is0(f) ? c : (o[a] = c, r.mul(c, f)), r.ONE), s = r.inv(i);
  return e.reduceRight((c, f, a) => r.is0(f) ? c : (o[a] = r.mul(c, o[a]), r.mul(c, f)), s), o;
}
function jt(t, e) {
  Rt(t);
  const n = t;
  be(n.ORDER, "FpLegendre");
  const r = (n.ORDER - T) / it, o = n.pow(e, r), i = n.eql(o, n.ONE), s = n.eql(o, n.ZERO), c = n.eql(o, n.neg(n.ONE));
  if (!i && !s && !c)
    throw new Error("invalid Legendre symbol result");
  return i ? 1 : s ? 0 : -1;
}
function Mo(t, e) {
  if (e !== void 0 && ln(e), t <= N)
    throw new Error("invalid n length: expected positive n, got " + t);
  if (e !== void 0 && e < 1)
    throw new Error("invalid n length: expected positive bit length, got " + e);
  const n = So(t);
  if (e !== void 0 && e < n)
    throw new Error(`invalid n length: expected nBitLength (${e}) >= bitLen(n) (${n})`);
  const r = e !== void 0 ? e : n, o = Math.ceil(r / 8);
  return { nBitLength: r, nByteLength: o };
}
const Ze = /* @__PURE__ */ new WeakMap();
class je {
  constructor(e, n = {}) {
    g(this, "ORDER");
    g(this, "BITS");
    g(this, "BYTES");
    g(this, "isLE");
    g(this, "ZERO", N);
    g(this, "ONE", T);
    g(this, "_lengths");
    g(this, "_mod");
    if (e <= T)
      throw new Error("invalid field: expected ORDER > 1, got " + e);
    let r;
    this.isLE = !1, n != null && typeof n == "object" && (typeof n.BITS == "number" && (r = n.BITS), typeof n.sqrt == "function" && Object.defineProperty(this, "sqrt", { value: n.sqrt, enumerable: !0 }), typeof n.isLE == "boolean" && (this.isLE = n.isLE), n.allowedLengths && (this._lengths = Object.freeze(n.allowedLengths.slice())), typeof n.modFromBytes == "boolean" && (this._mod = n.modFromBytes));
    const { nBitLength: o, nByteLength: i } = Mo(e, r);
    if (i > 2048)
      throw new Error("invalid field: expected ORDER of <= 2048 bytes");
    this.ORDER = e, this.BITS = o, this.BYTES = i, Object.freeze(this);
  }
  create(e) {
    return O(e, this.ORDER);
  }
  isValid(e) {
    if (typeof e != "bigint")
      throw new TypeError("invalid field element: expected bigint, got " + typeof e);
    return N <= e && e < this.ORDER;
  }
  is0(e) {
    return e === N;
  }
  // is valid and invertible
  isValidNot0(e) {
    return !this.is0(e) && this.isValid(e);
  }
  isOdd(e) {
    return (e & T) === T;
  }
  neg(e) {
    return O(-e, this.ORDER);
  }
  eql(e, n) {
    return e === n;
  }
  sqr(e) {
    return O(e * e, this.ORDER);
  }
  add(e, n) {
    return O(e + n, this.ORDER);
  }
  sub(e, n) {
    return O(e - n, this.ORDER);
  }
  mul(e, n) {
    return O(e * n, this.ORDER);
  }
  pow(e, n) {
    return No(e, n, this.ORDER);
  }
  div(e, n) {
    return O(e * Me(n, this.ORDER), this.ORDER);
  }
  // Same as above, but doesn't normalize
  sqrN(e) {
    return e * e;
  }
  addN(e, n) {
    return e + n;
  }
  subN(e, n) {
    return e - n;
  }
  mulN(e, n) {
    return e * n;
  }
  inv(e) {
    return Me(e, this.ORDER);
  }
  sqrt(e) {
    let n = Ze.get(this);
    return n || Ze.set(this, n = ko(this.ORDER)), n(this, e);
  }
  toBytes(e) {
    return this.isLE ? mn(e, this.BYTES) : yn(e, this.BYTES);
  }
  fromBytes(e, n = !1) {
    at(e);
    const { _lengths: r, BYTES: o, isLE: i, ORDER: s, _mod: c } = this;
    if (r) {
      if (e.length < 1 || !r.includes(e.length) || e.length > o)
        throw new Error("Field.fromBytes: expected " + r + " bytes, got " + e.length);
      const a = new Uint8Array(o);
      a.set(e, i ? 0 : a.length - e.length), e = a;
    }
    if (e.length !== o)
      throw new Error("Field.fromBytes: expected " + o + " bytes, got " + e.length);
    let f = i ? Mt(e) : wn(e);
    if (c && (f = O(f, s)), !n && !this.isValid(f))
      throw new Error("invalid field element: outside of range 0..ORDER");
    return f;
  }
  // TODO: we don't need it here, move out to separate fn
  invertBatch(e) {
    return Tn(this, e, !0);
  }
  // We can't move this out because Fp6, Fp12 implement it
  // and it's unclear what to return in there.
  cmov(e, n, r) {
    return Kt(r, "condition"), r ? n : e;
  }
}
function Ee(t, e = {}) {
  return Object.freeze(je.prototype), new je(t, e);
}
/*! noble-curves - MIT License (c) 2022 Paul Miller (paulmillr.com) */
const xe = /* @__PURE__ */ BigInt(0), Ot = /* @__PURE__ */ BigInt(1), Zo = /* @__PURE__ */ BigInt(4), re = 16, Pe = 128, jo = 5, Ye = 2 ** 31;
function Be(t) {
  const e = t;
  if (typeof e != "function")
    throw new TypeError('"Point" expected constructor, got type=' + typeof t);
  bt(e.fromAffine, "Point.fromAffine"), bt(e.fromBytes, "Point.fromBytes"), bt(e.fromHex, "Point.fromHex"), ut(e.BASE, "Point.BASE"), ut(e.ZERO, "Point.ZERO"), Rt(e.Fp), Rt(e.Fn);
}
function Po(t, e) {
  Be(t), _n(e, t);
  const n = Tn(t.Fp, e.map((r) => r.Z));
  return e.map((r, o) => t.fromAffine(r.toAffine(n[o])));
}
function Yo(t, e, n = 1) {
  if (!Number.isSafeInteger(t) || t < n || t > e)
    throw new Error("invalid window size, expected [" + n + ".." + e + "], got W=" + t);
}
function zo(t, e) {
  const n = t * (4 * e + 128);
  if (n > Ye)
    throw new Error("invalid window size: table would need ~" + Math.ceil(n / 2 ** 20) + " MiB, max " + Ye / 2 ** 20 + " MiB");
}
function Xo(t, e) {
  if (t !== void 0) {
    bt(t, "randomBytes");
    try {
      const n = t(e);
      if (!hn(n) || n.length !== e)
        return;
    } catch {
      return;
    }
    return t;
  }
}
function _n(t, e) {
  dn(t, "points"), t.forEach((n, r) => {
    if (!(n instanceof e))
      throw new Error("invalid point at index " + r);
  });
}
function Wo(t, e, n) {
  if (!Array.isArray(t))
    throw new Error("array of scalars expected");
  t.forEach((r, o) => {
    if (!(n === void 0 ? e.isValid(r) : ct(r) && r < n))
      throw new Error("invalid scalar at index " + o);
  });
}
const An = /* @__PURE__ */ new WeakMap();
function ie(t) {
  return An.get(t) || 1;
}
function Ko(t, e) {
  const n = t.double(), r = [t];
  for (let o = 1; o < e; o++)
    r.push(r[o - 1].add(n));
  return r;
}
function Vo(t, e) {
  const n = 2 ** e, r = n / 2, o = BigInt(n - 1), i = [];
  for (; t > xe; ) {
    let s = 0;
    t & Ot && (s = Number(t & o), s >= r && (s -= n), t -= BigInt(s)), i.push(s), t >>= Ot;
  }
  return i;
}
function Go(t, e, n) {
  const r = 2 ** e, o = r / 2, i = BigInt(r - 1), s = BigInt(e), c = [];
  for (let f = 0; f < n; f++) {
    let a = Number(t & i);
    t >>= s, a > o && (a -= r, t += Ot), c.push(a);
  }
  if (t !== xe)
    throw new Error("invalid wnaf");
  return c;
}
function Jo(t, e, n) {
  let r = 0;
  for (const i of n)
    r = Math.max(r, i.length);
  let o = t;
  for (let i = r - 1; i >= 0; i--) {
    i !== r - 1 && (o = o.double());
    for (let s = 0; s < n.length; s++) {
      const c = n[s][i];
      if (c) {
        const f = e[s][Math.abs(c) - 1 >> 1];
        o = o.add(c < 0 ? f.negate() : f);
      }
    }
  }
  return o;
}
class Qo {
  // Parametrized with a given Point class (not individual point)
  constructor(e, n) {
    g(this, "Point");
    g(this, "BASE");
    g(this, "ZERO");
    g(this, "randomBytes");
    g(this, "wnafPrecomputes", /* @__PURE__ */ new WeakMap());
    g(this, "baseCanBeBlinded");
    g(this, "bits");
    Be(e), this.randomBytes = Xo(n, re), this.Point = e, this.BASE = e.BASE, this.ZERO = e.ZERO, this.bits = e.Fn.BITS;
  }
  /**
   * Creates a signed fixed-window wNAF precomputation table: for every window w, the
   * multiples `[1..2^(W−1)]⋅2^(w⋅W)⋅P`, flattened. All doublings are baked into the table,
   * so cached multiplication is additions-only. `windows = ceil(bits/W) + 1`: the extra
   * window absorbs the final carry of signed-digit recoding.
   * For a 256-bit curve and W=6, the table is 44⋅32 = 1408 points.
   * @param point - Point instance
   * @param W - window size
   * @param bits - scalar bitlength the table must cover
   */
  buildWnafTable(e, n, r) {
    const o = Math.ceil(r / n) + 1, i = 2 ** (n - 1), s = [];
    let c = e;
    for (let f = 0; f < o; f++) {
      let a = c;
      for (let u = 0; u < i; u++)
        s.push(a), a = a.add(c);
      c = s[s.length - 1].double();
    }
    return { W: n, bits: r, windows: o, comp: s };
  }
  /**
   * Implements ec multiplication using precomputed signed fixed-window wNAF tables.
   * Constant-time: fixed window count with one table addition per window — zero digits feed
   * the fake accumulator — and no doublings; the lookup scans the whole window slice.
   * Scalar bounds are validated by the public entry points ({@link ScalarMultiplier.mulCT},
   * {@link ScalarMultiplier.mulCTBlinded}, {@link ScalarMultiplier.mulUnsafe});
   * signedWindowDigits throws if `n` exceeds the table.
   * @returns real and fake (for const-time) points
   */
  wnafCachedCT(e, n) {
    const { W: r, windows: o, comp: i } = e, s = 2 ** (r - 1), c = Go(n, r, o);
    let f = this.ZERO, a = this.BASE;
    for (let u = 0; u < o; u++) {
      const d = c[u], l = u * s, h = Math.abs(d) - 1;
      let w = i[l];
      for (let x = 1; x < s; x++)
        w = x === h ? i[l + x] : w;
      const m = w.negate();
      d === 0 ? a = a.add(i[l]) : f = f.add(d < 0 ? m : w);
    }
    return { p: f, f: a };
  }
  // Cache key is point identity plus (W, bits); at most two entries exist per point (public-width
  // `Fn.BITS` and blinded `Fn.BITS + BLIND_BITS`). Callers must not reuse the same point with
  // incompatible `transform(...)` layouts and expect a separate cache entry.
  getWnafPrecomputes(e, n, r, o) {
    let i = this.wnafPrecomputes.get(n), s = i == null ? void 0 : i.find((c) => c.W === e && c.bits === r);
    return s || (s = this.buildWnafTable(n, e, r), typeof o == "function" && (s = { ...s, comp: o(s.comp) }), i || (i = [], this.wnafPrecomputes.set(n, i)), i.push(s)), s;
  }
  assertPoint(e) {
    if (!(e instanceof this.Point))
      throw new TypeError('"point" expected Point instance, got type=' + typeof e);
  }
  // Shared prologue of the constant-time entry points. Rejects scalar 0: in key/signature-style
  // callers a zero scalar means broken upstream plumbing, and concrete Points already reject it.
  // Uses inRange instead of Fn.isValidNot0: validateField() only certifies the arithmetic subset.
  validateMulInput(e, n) {
    if (this.assertPoint(e), !bn(n, Ot, this.Point.Fn.ORDER))
      throw new Error("invalid scalar");
  }
  // Constant-time dispatch shared by mulCT / mulCTBlinded. Un-precomputed points (W===1, e.g.
  // ECDH peer keys) skip building a throwaway cached table in favor of a small fixed-window
  // multiply. `n` must be < 2^bits.
  runCT(e, n, r, o) {
    const i = ie(e);
    return i === 1 ? this.fixedWindowCT(e, n, r) : this.wnafCachedCT(this.getWnafPrecomputes(i, e, r, o), n);
  }
  mulCT(e, n, r) {
    return this.validateMulInput(e, n), this.runCT(e, n, this.bits, r);
  }
  mulCTBlinded(e, n, r) {
    if (this.validateMulInput(e, n), this.randomBytes === void 0)
      throw new Error("randomBytes is required for scalar blinding");
    const o = this.Point.Fn.BITS + Pe, i = this.randomBytes(re);
    if (!hn(i) || i.length !== re)
      throw new Error("randomBytes returned invalid byte array");
    i[0] = i[0] & 63 | 128;
    const s = n + wn(i) * this.Point.Fn.ORDER;
    return this.runCT(e, s, o, r);
  }
  /**
   * Constant-time multiplication `n*point` for an un-precomputed point, via a small fixed window.
   * A cached wNAF table only pays off when reused; a flat 2^FW_WINDOW table (`size-1` adds) is
   * far cheaper to build for a single use. The point-operation sequence is independent of `n`:
   * build the table, then per window exactly FW_WINDOW doublings, a data-oblivious scan over
   * every table entry, and one addition (adds the identity when the window digit is 0 — never
   * skipped).
   *
   * `n` must be `< 2^bits`. Assumes complete addition (adding the identity costs the same as any
   * add), which holds for the Weierstrass/Edwards point types used here. The table is left in
   * projective form (no normalizeZ): normalizing this small a table costs more than the
   * mixed-add savings it would buy for a single multiply.
   * @returns real point `p`; `f` duplicates it only to match {@link wnafCachedCT}'s return shape
   * (this path needs no fake accumulator — its op-count is already scalar-independent).
   */
  fixedWindowCT(e, n, r) {
    const o = jo, i = 1 << o, s = Io(o), c = new Array(i);
    c[0] = this.ZERO;
    for (let u = 1; u < i; u++)
      c[u] = c[u - 1].add(e);
    const f = Math.ceil(r / o);
    let a = this.ZERO;
    for (let u = f - 1; u >= 0; u--) {
      if (u !== f - 1)
        for (let h = 0; h < o; h++)
          a = a.double();
      const d = Number(n >> BigInt(u * o) & s);
      let l = c[0];
      for (let h = 1; h < i; h++)
        l = h === d ? c[h] : l;
      a = a.add(l);
    }
    return { p: a, f: a };
  }
  shouldBlind(e, n) {
    return this.randomBytes === void 0 ? !1 : n === Ot ? !0 : e !== this.BASE ? !1 : (this.baseCanBeBlinded === void 0 && (this.baseCanBeBlinded = this.mulUnsafe(this.BASE, this.Point.Fn.ORDER).is0()), this.baseCanBeBlinded);
  }
  mulSecret(e, n, r, o) {
    return this.shouldBlind(e, r) ? this.mulCTBlinded(e, n, o) : this.mulCT(e, n, o);
  }
  mulUnsafe(e, n, r) {
    if (this.assertPoint(e), !ct(n))
      throw new Error("invalid scalar");
    const o = ie(e);
    if (o === 1 || n >= this.Point.Fn.ORDER)
      return tr(this.Point, [e], [n], !0);
    const i = this.getWnafPrecomputes(o, e, this.bits, r);
    return this.wnafCachedCT(i, n).p;
  }
  // Remembers the window size used for precomputed wNAF multiplication of the given point
  // and drops any previously built tables. Usually only the base point is precomputed.
  // W=1 resets the point to the un-precomputed (table-less) paths.
  // W is additionally capped so tables stay under ~2 GiB ({@link TABLE_BYTES_MAX}).
  setWindowSize(e, n) {
    this.assertPoint(e), Yo(n, this.bits);
    const r = Math.ceil((this.bits + Pe) / n) + 1;
    zo(r * 2 ** (n - 1), this.Point.Fp.BYTES), An.set(e, n), this.wnafPrecomputes.delete(e);
  }
  // True when a window size is set: tables themselves are built lazily on first multiply.
  hasWindowSize(e) {
    return ie(e) !== 1;
  }
}
function tr(t, e, n, r = !1) {
  if (Be(t), _n(e, t), Kt(r, "allowOversized"), Wo(n, t.Fn, r ? t.Fn.ORDER ** Zo : void 0), e.length !== n.length)
    throw new Error("arrays of points and scalars must have equal length");
  const o = e.map((s) => Ko(s, 4)), i = n.map((s) => Vo(s, 4));
  return Jo(t.ZERO, o, i);
}
function ze(t, e, n) {
  if (e) {
    if (e.ORDER !== t)
      throw new Error("Field.ORDER must match order: Fp == p, Fn == n");
    return Rt(e), e;
  } else
    return Ee(t, { isLE: n });
}
function er(t, e, n = {}, r) {
  if (r === void 0 && (r = t === "edwards"), !e || typeof e != "object")
    throw new Error(`expected valid ${t} CURVE object`);
  Zt(n);
  for (const f of ["p", "n", "h"]) {
    const a = e[f];
    if (!(ct(a) && a !== xe))
      throw new Error(`CURVE.${f} must be positive bigint`);
  }
  const o = ze(e.p, n.Fp, r), i = ze(e.n, n.Fn, r), c = ["Gx", "Gy", "a", "d"];
  for (const f of c)
    if (!o.isValid(e[f]))
      throw new Error(`CURVE.${f} must be valid field element of CURVE.Fp`);
  return e = Object.freeze(Object.assign({}, e)), { CURVE: e, Fp: o, Fn: i };
}
function nr(t, e) {
  return function(r) {
    const o = t(r);
    return { secretKey: o, publicKey: e(o) };
  };
}
/*! noble-curves - MIT License (c) 2022 Paul Miller (paulmillr.com) */
const Ut = /* @__PURE__ */ BigInt(0), $t = /* @__PURE__ */ BigInt(1), Nt = /* @__PURE__ */ BigInt(2), or = /* @__PURE__ */ BigInt(4), Xe = /* @__PURE__ */ BigInt(8);
function rr(t, e, n, r) {
  const o = t.sqr(n), i = t.sqr(r), s = t.add(t.mul(e.a, o), i), c = t.add(t.ONE, t.mul(e.d, t.mul(o, i)));
  return t.eql(s, c);
}
function ir(t, e = {}) {
  Zt(e, {}, {}, "extraOpts");
  const n = e, r = er("edwards", t, n, n.FpFnLE), { Fp: o, Fn: i } = r;
  let s = r.CURVE;
  const { h: c } = s;
  if (jt(o, s.a) !== 1)
    throw new Error("edwards: CURVE.a must be a square in Fp for complete addition formulas");
  if (jt(o, s.d) !== -1)
    throw new Error("edwards: CURVE.d must be a non-square in Fp for complete addition formulas");
  Zt(n, {}, { uvRatio: "function", randomBytes: "function" });
  const f = n.randomBytes === void 0 ? pn : n.randomBytes, a = Nt << BigInt(o.BYTES * 8) - $t;
  function u(R) {
    if (!o.isOdd)
      throw new Error("Field does not have .isOdd()");
    return o.isOdd(R);
  }
  const d = n.uvRatio === void 0 ? (R, p) => {
    try {
      return { isValid: !0, value: o.sqrt(o.div(R, p)) };
    } catch {
      return { isValid: !1, value: Ut };
    }
  } : n.uvRatio;
  if (!rr(o, s, s.Gx, s.Gy))
    throw new Error("bad curve params: generator point");
  const l = o.eql(s.a, o.neg(o.ONE)) ? (R) => o.neg(R) : o.eql(s.a, o.ONE) ? (R) => R : (R) => o.mul(s.a, R);
  function h(R, p, E = !1) {
    const b = E ? $t : Ut;
    return Et("coordinate " + R, p, b, a), p;
  }
  function w(R) {
    if (!(R instanceof m))
      throw new Error("EdwardsPoint expected");
  }
  const y = class y {
    constructor(p, E, b, A) {
      g(this, "X");
      g(this, "Y");
      g(this, "Z");
      g(this, "T");
      this.X = h("x", p), this.Y = h("y", E), this.Z = h("z", b, !0), this.T = h("t", A), Object.freeze(this);
    }
    static CURVE() {
      return s;
    }
    /**
     * Create one extended Edwards point from affine coordinates.
     * Does NOT validate that the point is on-curve or torsion-free.
     * Use `.assertValidity()` on adversarial inputs.
     */
    static fromAffine(p) {
      if (p instanceof y)
        throw new Error("extended point not allowed");
      const { x: E, y: b } = p || {};
      return h("x", E), h("y", b), new y(E, b, o.ONE, o.mul(E, b));
    }
    // Uses algo from RFC8032 5.1.3.
    static fromBytes(p, E = !1) {
      const b = o.BYTES, { a: A, d: S } = s;
      p = vt(at(p, b, "point")), Kt(E, "zip215");
      const L = vt(p), C = p[b - 1];
      L[b - 1] = C & -129;
      const q = Mt(L), k = E ? a : o.ORDER;
      Et("point.y", q, Ut, k);
      const U = o.sqr(q), F = o.sub(U, o.ONE), j = o.sub(o.mulN(S, U), A);
      let { isValid: B, value: _ } = d(F, j);
      if (!B)
        throw new Error("bad point: invalid y coordinate");
      const $ = u(_), M = (C & 128) !== 0;
      if (!E && o.is0(_) && M)
        throw new Error("bad point: x=0 and x_0=1");
      return M !== $ && (_ = o.neg(_)), y.fromAffine({ x: _, y: q });
    }
    static fromHex(p, E = !1) {
      return y.fromBytes(To(p), E);
    }
    get x() {
      return this.toAffine().x;
    }
    get y() {
      return this.toAffine().y;
    }
    precompute(p = 6, E = !0) {
      return v.setWindowSize(this, p), E || this.multiply(Nt), this;
    }
    // Useful in fromAffine() - not for fromBytes(), which always created valid points.
    assertValidity() {
      const p = this, { a: E, d: b } = s;
      if (p.is0())
        throw new Error("bad point: ZERO");
      const { X: A, Y: S, Z: L, T: C } = p, q = o.sqr(A), k = o.sqr(S), U = o.sqr(L), F = o.sqr(U), j = o.mul(q, E), B = o.mul(o.add(j, k), U), _ = o.add(F, o.mul(b, o.mul(q, k)));
      if (!o.eql(B, _))
        throw new Error("bad point: equation left != right (1)");
      const $ = o.mul(A, S), M = o.mul(L, C);
      if (!o.eql($, M))
        throw new Error("bad point: equation left != right (2)");
    }
    // Compare one point to another.
    equals(p) {
      w(p);
      const { X: E, Y: b, Z: A } = this, { X: S, Y: L, Z: C } = p, q = o.mul(E, C), k = o.mul(S, A), U = o.mul(b, C), F = o.mul(L, A);
      return o.eql(q, k) && o.eql(U, F);
    }
    is0() {
      return this.equals(y.ZERO);
    }
    negate() {
      return new y(o.neg(this.X), this.Y, this.Z, o.neg(this.T));
    }
    // Fast algo for doubling Extended Point.
    // https://hyperelliptic.org/EFD/g1p/auto-twisted-extended.html#doubling-dbl-2008-hwcd
    // Cost: 4M + 4S + 1*a + 6add + 1*2.
    double() {
      const { X: p, Y: E, Z: b } = this, A = o.sqr(p), S = o.sqr(E), L = o.mul(o.sqr(b), Nt), C = l(A), q = o.addN(p, E), k = o.sub(o.subN(o.sqr(q), A), S), U = o.addN(C, S), F = o.subN(U, L), j = o.subN(C, S), B = o.mul(k, F), _ = o.mul(U, j), $ = o.mul(k, j), M = o.mul(F, U);
      return new y(B, _, M, $);
    }
    // Fast algo for adding 2 Extended Points.
    // https://hyperelliptic.org/EFD/g1p/auto-twisted-extended.html#addition-add-2008-hwcd
    // Cost: 9M + 1*a + 1*d + 7add.
    add(p) {
      w(p);
      const { d: E } = s, { X: b, Y: A, Z: S, T: L } = this, { X: C, Y: q, Z: k, T: U } = p, F = o.mul(b, C), j = o.mul(A, q), B = o.mul(o.mulN(L, E), U), _ = o.mul(S, k), $ = o.sub(o.subN(o.mulN(o.addN(b, A), o.addN(C, q)), F), j), M = o.subN(_, B), Y = o.addN(_, B), z = o.sub(j, l(F)), G = o.mul($, M), J = o.mul(Y, z), Gt = o.mul($, z), At = o.mul(M, Y);
      return new y(G, J, At, Gt);
    }
    subtract(p) {
      return w(p), this.add(p.negate());
    }
    // Constant-time multiplication.
    multiply(p) {
      if (!i.isValidNot0(p))
        throw new RangeError("invalid scalar: expected 1 <= sc < curve.n");
      const { p: E, f: b } = v.mulSecret(this, p, c, x);
      return x([E, b])[0];
    }
    // Non-constant-time multiplication. Uses double-and-add algorithm.
    // It's faster, but should only be used when you don't care about
    // an exposed private key e.g. sig verification.
    // Keeps the same subgroup-scalar contract: 0 is allowed for public-scalar callers, but
    // n and larger values are rejected instead of being reduced mod n to the identity point.
    multiplyUnsafe(p) {
      if (!i.isValid(p))
        throw new RangeError("invalid scalar: expected 0 <= sc < curve.n");
      return p === Ut ? y.ZERO : this.is0() || p === $t ? this : v.mulUnsafe(this, p, x);
    }
    // Checks if point is of small order.
    // If you add something to small order point, you will have "dirty"
    // point with torsion component.
    // Clears cofactor and checks if the result is 0.
    isSmallOrder() {
      return this.clearCofactor().is0();
    }
    // Multiplies point by curve order and checks if the result is 0.
    // Returns `false` is the point is dirty.
    isTorsionFree() {
      return v.mulUnsafe(this, s.n).is0();
    }
    // Converts Extended point to default (x, y) coordinates.
    // Can accept precomputed Z^-1 - for example, from invertBatch.
    toAffine(p) {
      const E = this;
      let b = p;
      if (b != null && typeof b != "bigint")
        throw new TypeError('"invertedZ" expected bigint, got type=' + typeof b);
      const { X: A, Y: S, Z: L } = E, C = E.is0();
      b == null && (b = C ? o.create(Xe) : o.inv(L));
      const q = o.mul(A, b), k = o.mul(S, b), U = o.mul(L, b);
      if (C)
        return { x: o.ZERO, y: o.ONE };
      if (!o.eql(U, o.ONE))
        throw new Error("invZ was invalid");
      return { x: q, y: k };
    }
    clearCofactor() {
      return c === $t ? this : c === Nt ? this.double() : c === or ? this.double().double() : c === Xe ? this.double().double().double() : this.multiplyUnsafe(c);
    }
    toBytes() {
      const { x: p, y: E } = this.toAffine(), b = o.toBytes(E);
      return b[b.length - 1] |= u(p) ? 128 : 0, b;
    }
    toHex() {
      return Oo(this.toBytes());
    }
    toString() {
      return `<Point ${this.is0() ? "ZERO" : this.toHex()}>`;
    }
  };
  g(y, "BASE", new y(s.Gx, s.Gy, o.ONE, o.mul(s.Gx, s.Gy))), g(y, "ZERO", new y(o.ZERO, o.ONE, o.ONE, o.ZERO)), g(y, "Fp", o), g(y, "Fn", i);
  let m = y;
  const x = (R) => Po(m, R), v = new Qo(m, f);
  return v.bits >= 6 && m.BASE.precompute(6), Object.freeze(m.prototype), Object.freeze(m), m;
}
/*! noble-curves - MIT License (c) 2022 Paul Miller (paulmillr.com) */
const ot = /* @__PURE__ */ BigInt(0), P = /* @__PURE__ */ BigInt(1), qt = /* @__PURE__ */ BigInt(2);
function We(t, e) {
  return t + e - (e >> P << P);
}
function sr(t) {
  const e = BigInt(6) * t;
  return (n, r, o) => {
    const i = r + o, c = ((e + o - r) * n + r) % t;
    return { x_2: c, x_3: i - c };
  };
}
function cr(t) {
  return Zt(t, {
    P: "bigint",
    type: "string",
    adjustScalarBytes: "function",
    powPminus2: "function"
  }, {
    randomBytes: "function",
    scalarMultBase: "function"
  }), Object.freeze({ ...t });
}
function fr(t) {
  const e = cr(t), { P: n, type: r, adjustScalarBytes: o, powPminus2: i, randomBytes: s } = e, c = e.scalarMultBase, f = r === "x25519";
  if (!f && r !== "x448")
    throw new Error("invalid type");
  const a = s === void 0 ? pn : s, u = f ? 255 : 448, d = sr(n), l = f ? 32 : 56, h = BigInt(f ? 9 : 5), w = BigInt(f ? 121665 : 39081), m = f ? qt ** BigInt(254) : qt ** BigInt(447), x = f ? BigInt(8) * (qt ** BigInt(251) - P) : BigInt(4) * (qt ** BigInt(445) - P), v = m + x + P, y = (B) => O(B, n), R = p(h);
  function p(B) {
    return mn(y(B), l);
  }
  function E(B) {
    const _ = vt(at(B, l, "uCoordinate"));
    return f && (_[31] &= 127), y(Mt(_));
  }
  function b(B) {
    return Mt(o(vt(at(B, l, "scalar"))));
  }
  const A = new Set(f ? [
    ot,
    P,
    n - P,
    BigInt("325606250916557431795983626356110631294008115727848805560023387167927233504"),
    BigInt("39382357235489614581723060781553021112529911719440698176882885853963445705823")
  ] : [ot, P, n - P]);
  function S(B, _) {
    const $ = E(_);
    if (A.has($))
      throw new Error("invalid private or public key received");
    const M = k($, b(B));
    if (M === ot)
      throw new Error("invalid private or public key received");
    return p(M);
  }
  function L(B) {
    if (c === void 0)
      return S(B, R);
    const _ = b(B);
    Et("scalar", _, m, v);
    const $ = y(c(_));
    if ($ === ot)
      throw new Error("invalid private or public key received");
    return p($);
  }
  const C = L, q = S;
  function k(B, _) {
    Et("u", B, ot, n), Et("scalar", _, m, v);
    const $ = _, M = B;
    let Y = P, z = ot, G = B, J = P;
    const Gt = $ ^ $ >> P;
    for (let Jt = BigInt(u - 1); Jt >= ot; Jt--) {
      const Te = We(n, Gt >> Jt);
      ({ x_2: Y, x_3: G } = d(Te, Y, G)), { x_2: z, x_3: J } = d(Te, z, J);
      const Qt = Y + z, te = y(Qt * Qt), ee = Y - z, _e = y(ee * ee), Ae = te - _e, Hn = G + J, kn = G - J, Se = y(kn * Qt), Ie = y(Hn * ee), Le = Se + Ie, Ce = Se - Ie;
      G = y(Le * Le), J = y(M * y(Ce * Ce)), Y = y(te * _e), z = y(Ae * (te + y(w * Ae)));
    }
    const At = We(n, $);
    ({ x_2: Y, x_3: G } = d(At, Y, G)), { x_2: z, x_3: J } = d(At, z, J);
    const qn = i(z);
    return y(Y * qn);
  }
  const U = {
    secretKey: l,
    publicKey: l,
    seed: l
  }, F = (B) => (B = B === void 0 ? a(l) : B, at(B, U.seed, "seed"), B), j = { randomSecretKey: F };
  return Object.freeze(U), Object.freeze(j), Object.freeze({
    keygen: nr(F, C),
    getSharedSecret: q,
    getPublicKey: C,
    scalarMult: S,
    scalarMultBase: L,
    utils: j,
    GuBytes: R.slice(),
    lengths: U
  });
}
/*! noble-curves - MIT License (c) 2022 Paul Miller (paulmillr.com) */
const Ke = /* @__PURE__ */ BigInt(0), ar = /* @__PURE__ */ BigInt(1), Ve = /* @__PURE__ */ BigInt(2), ur = /* @__PURE__ */ BigInt(3), dr = /* @__PURE__ */ BigInt(5), lr = /* @__PURE__ */ BigInt(8), Vt = /* @__PURE__ */ BigInt("0x7fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffed"), hr = {
  p: Vt,
  n: BigInt("0x1000000000000000000000000000000014def9dea2f79cd65812631a5cf5d3ed"),
  h: lr,
  a: BigInt("0x7fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffec"),
  d: BigInt("0x52036cee2b6ffe738cc740797779e89800700a4d4141d8ab75eb4dca135978a3"),
  Gx: BigInt("0x216936d3cd6e53fec0a4e231fdd6dc5c692cc7609525a7b2c9562d608f25d51a"),
  Gy: BigInt("0x6666666666666666666666666666666666666666666666666666666666666658")
};
function Sn(t) {
  const e = BigInt(10), n = BigInt(20), r = BigInt(40), o = BigInt(80), i = Vt, c = t * t % i * t % i, f = X(c, Ve, i) * c % i, a = X(f, ar, i) * t % i, u = X(a, dr, i) * a % i, d = X(u, e, i) * u % i, l = X(d, n, i) * d % i, h = X(l, r, i) * l % i, w = X(h, o, i) * h % i, m = X(w, o, i) * h % i, x = X(m, e, i) * u % i;
  return { pow_p_5_8: X(x, Ve, i) * t % i, b2: c };
}
function pr(t) {
  return t[0] &= 248, t[31] &= 127, t[31] |= 64, t;
}
const Ge = /* @__PURE__ */ BigInt("19681161376707505956807079304988542015446066515923890162744021073123829784752");
function gr(t, e) {
  const n = Vt, r = O(e * e * e, n), o = O(r * r * e, n), i = Sn(t * o).pow_p_5_8;
  let s = O(t * r * i, n);
  const c = O(e * s * s, n), f = s, a = O(s * Ge, n), u = c === t, d = c === O(-t, n), l = c === O(-t * Ge, n);
  return u && (s = f), (d || l) && (s = a), Do(s, n) && (s = O(-s, n)), { isValid: u || d, value: s };
}
const Je = /* @__PURE__ */ ir(hr, { uvRatio: gr }), Qe = /* @__PURE__ */ (() => {
  const t = Vt, e = (n) => {
    const { pow_p_5_8: r, b2: o } = Sn(n);
    return O(X(r, ur, t) * o, t);
  };
  return fr({
    P: t,
    type: "x25519",
    powPminus2: e,
    adjustScalarBytes: pr,
    // ~3x faster fixed-base: [k]B on the birationally-equivalent Edwards curve using cached
    // base tables, mapped back via u = (1+y)/(1-y) = (Z+Y)/(Z-Y) with one Fermat inversion.
    // Same construction as libsodium's crypto_scalarmult_curve25519_base.
    scalarMultBase: (n) => {
      const r = O(n, Je.Fn.ORDER);
      if (r === Ke)
        return Ke;
      const o = Je.BASE.multiply(r);
      return O((o.Z + o.Y) * e(O(o.Z - o.Y, t)), t);
    }
  });
})();
class tn {
  constructor(e, n) {
    g(this, "oHash");
    g(this, "iHash");
    g(this, "blockLen");
    g(this, "outputLen");
    g(this, "canXOF", !1);
    g(this, "finished", !1);
    g(this, "destroyed", !1);
    if (we(e), Q(n, void 0, "key"), this.iHash = e.create(), typeof this.iHash.update != "function")
      throw new Error("expected Hash instance");
    this.blockLen = this.iHash.blockLen, this.outputLen = this.iHash.outputLen;
    const r = this.blockLen, o = new Uint8Array(r);
    o.set(n.length > r ? e.create().update(n).digest() : n);
    for (let i = 0; i < o.length; i++)
      o[i] ^= 54;
    this.iHash.update(o), this.oHash = e.create();
    for (let i = 0; i < o.length; i++)
      o[i] ^= 106;
    this.oHash.update(o), rt(o);
  }
  update(e) {
    return Dt(this), this.iHash.update(e), this;
  }
  digestInto(e) {
    Dt(this), fn(e, this), this.finished = !0;
    const n = e.subarray(0, this.outputLen);
    this.iHash.digestInto(n), this.oHash.update(n), this.oHash.digestInto(n), this.destroy();
  }
  digest() {
    const e = new Uint8Array(this.oHash.outputLen);
    return this.digestInto(e), e;
  }
  _cloneInto(e) {
    e || (e = Object.create(Object.getPrototypeOf(this), {}));
    const { oHash: n, iHash: r, finished: o, destroyed: i, blockLen: s, outputLen: c, canXOF: f } = this;
    return e = e, e.finished = o, e.destroyed = i, e.blockLen = s, e.outputLen = c, e.canXOF = f, e.oHash = n._cloneInto(e.oHash), e.iHash = r._cloneInto(e.iHash), e;
  }
  clone() {
    return this._cloneInto();
  }
  destroy() {
    this.destroyed = !0, this.oHash.destroy(), this.iHash.destroy();
  }
}
const Tt = /* @__PURE__ */ (() => {
  const t = ((e, n, r) => new tn(e, n).update(r).digest());
  return t.create = (e, n) => new tn(e, n), t;
})(), gt = /* @__PURE__ */ Uint8Array.of(0), wr = /* @__PURE__ */ Uint8Array.of();
function yr(t, e, n, r = 32, o) {
  we(t), ft(r, "length"), Q(e, void 0, "prk");
  const i = t.outputLen;
  if (e.length < i)
    throw new Error('"prk" must be at least HashLen octets');
  if (r > 255 * i)
    throw new Error("Length must be <= 255*HashLen");
  const s = Math.ceil(r / i);
  if (n === void 0 ? n = wr : Q(n, void 0, "info"), !s)
    return o && rt(e), new Uint8Array();
  const c = o && s === 1 ? e : new Uint8Array(s * i), { iHash: f, oHash: a } = Tt.create(t, e), u = o ? e : new Uint8Array(i), d = s > 1 ? (o == null ? void 0 : o.iHash) || t.create() : void 0;
  for (let h = 0; h < s - 1; h++) {
    gt[0] = h + 1;
    const w = f._cloneInto(d);
    h && w.update(u), w.update(n).update(gt).digestInto(u), a._cloneInto(d).update(u).digestInto(u), c.set(u, i * h);
  }
  if (gt[0] = s, s > 1 && f.update(u), f.update(n).update(gt).digestInto(u), a.update(u).digestInto(u), c.set(u, i * (s - 1)), f.destroy(), a.destroy(), d == null || d.destroy(), u !== c && rt(u), rt(gt), r === c.length)
    return c;
  const l = c.slice(0, r);
  return rt(c), l;
}
const mr = (t, e, n, r, o) => {
  we(t), n === void 0 && (n = new Uint8Array(t.outputLen));
  const i = Tt.create(t, n).update(e);
  return yr(t, i.digest(), r, o, i);
}, br = {
  waiting: "Waiting on your computer",
  running: "Running",
  done: "Done",
  failed: "Did not finish",
  dismissed: "Dismissed on the computer"
}, V = new TextEncoder(), ve = new TextDecoder();
function D(t) {
  const e = document.querySelector(t);
  if (!e) throw new Error(`Air Handoff receiver is missing ${t}`);
  return e;
}
const Er = D("#pair-card"), xr = D("#receive-card"), yt = D("#pair-button"), Br = D("#forget-button"), In = D("#device-name"), Ht = D("#pair-error"), en = D("#receive-error"), nn = D("#connection-state"), Ln = D("#empty-state"), Cn = D("#transfer-list"), Un = D("#ask-card"), le = D("#ask-text"), Pt = D("#ask-button"), dt = D("#ask-error"), vr = D("#request-list");
let Z = Or(), he = 0;
const lt = /* @__PURE__ */ new Set();
In.value = navigator.userAgent.includes("Android") ? "Android phone" : navigator.userAgent.includes("iPhone") ? "iPhone" : "My phone";
function ht(t) {
  let e = "";
  for (const n of t) e += String.fromCharCode(n);
  return btoa(e).replaceAll("+", "-").replaceAll("/", "_").replaceAll("=", "");
}
function W(t) {
  const e = t.replaceAll("-", "+").replaceAll("_", "/") + "=".repeat((4 - t.length % 4) % 4);
  return Uint8Array.from(atob(e), (n) => n.charCodeAt(0));
}
function Rr(t, e) {
  if (t.length !== e.length) return !1;
  let n = 0;
  for (let r = 0; r < t.length; r += 1)
    n |= t[r] ^ e[r];
  return n === 0;
}
function Or() {
  try {
    const t = JSON.parse(localStorage.getItem("heliox_air_handoff_credential") || "null");
    if (t != null && t.device_id && (t != null && t.device_secret)) return t;
  } catch {
  }
  return null;
}
function Tr(t) {
  Z = t, localStorage.setItem("heliox_air_handoff_credential", JSON.stringify(t));
}
function _r() {
  Z = null, localStorage.removeItem("heliox_air_handoff_credential"), lt.clear(), Cn.replaceChildren(), Oe();
}
async function Ar() {
  Ht.textContent = "";
  const e = new URLSearchParams(location.hash.slice(1)).get("pair");
  if (!e) {
    Ht.textContent = "Open this page by scanning the current pairing QR in Heliox.";
    return;
  }
  const n = In.value.trim();
  if (!n) {
    Ht.textContent = "Enter a name for this phone.";
    return;
  }
  yt.disabled = !0, yt.textContent = "Establishing encrypted channel…";
  try {
    const r = W(e), o = Qe.keygen(), i = Tt(mt, r, De(V.encode("pair-v1:"), o.publicKey)), s = await fetch("/api/pair", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        device_name: n,
        client_public_key: ht(o.publicKey),
        client_proof: ht(i)
      })
    }), c = await s.json();
    if (!s.ok) throw new Error(c.error || "Pairing failed");
    const f = W(c.server_public_key), a = Tt(
      mt,
      r,
      De(V.encode("server-v1:"), f, o.publicKey)
    );
    if (!Rr(a, W(c.server_proof)))
      throw new Error("The computer could not be authenticated");
    const u = Qe.getSharedSecret(o.secretKey, f), d = mr(mt, u, r, V.encode("heliox-air-handoff-pair-v1"), 32), l = Xt(d, W(c.nonce), V.encode("heliox-air-handoff-credential-v1")).decrypt(
      W(c.credential)
    );
    Tr(JSON.parse(ve.decode(l))), history.replaceState(null, "", location.pathname), Oe(), await Re();
  } catch (r) {
    Ht.textContent = r instanceof Error ? r.message : "Pairing failed";
  } finally {
    yt.disabled = !1, yt.textContent = "Pair securely";
  }
}
async function _t(t, e = {}) {
  if (!Z) throw new Error("This phone is not paired");
  const n = (e.method || "GET").toUpperCase(), r = typeof e.body == "string" ? V.encode(e.body) : new Uint8Array(), o = String(Date.now() / 1e3), i = ht(rn(18)), s = V.encode([n, t, o, i, Wt(mt(r))].join(`
`)), c = Tt(mt, W(Z.device_secret), s);
  return fetch(t, {
    ...e,
    headers: {
      ...e.headers || {},
      "X-Heliox-Device": Z.device_id,
      "X-Heliox-Time": o,
      "X-Heliox-Nonce": i,
      "X-Heliox-Signature": ht(c)
    }
  });
}
function $n(t, e) {
  if (!Z) throw new Error("This phone is not paired");
  const n = Xt(W(Z.device_secret), W(t.nonce), V.encode(e)).decrypt(
    W(t.ciphertext)
  );
  return JSON.parse(ve.decode(n));
}
function Sr(t, e) {
  if (!Z) throw new Error("This phone is not paired");
  const n = rn(12), r = Xt(W(Z.device_secret), n, V.encode(e)).encrypt(
    V.encode(JSON.stringify(t))
  );
  return { nonce: ht(n), ciphertext: ht(r) };
}
async function Nn() {
  const t = await _t("/api/requests"), e = await t.json();
  if (!t.ok) throw new Error(e.error || "Could not read your requests");
  const n = $n(e, "requests-v1");
  Un.classList.toggle("hidden", !n.accepting && n.requests.length === 0), Pt.disabled = !n.accepting, le.disabled = !n.accepting, n.accepting ? dt.textContent === "Requests are turned off on your computer." && (dt.textContent = "") : dt.textContent = "Requests are turned off on your computer.";
  const r = n.requests.map((o) => {
    const i = document.createElement("li");
    i.textContent = o.text;
    const s = document.createElement("span");
    if (s.textContent = br[o.status] ?? o.status, i.append(s), o.result) {
      const c = document.createElement("p");
      c.className = "answer", c.textContent = o.result, i.append(c);
    }
    return i;
  });
  vr.replaceChildren(...r);
}
async function Ir() {
  const t = le.value.trim();
  if (dt.textContent = "", !t) {
    dt.textContent = "Type what you want Heliox to do.";
    return;
  }
  Pt.disabled = !0;
  try {
    const e = await _t("/api/requests", {
      method: "POST",
      body: JSON.stringify(Sr({ text: t }, "request-v1")),
      headers: { "Content-Type": "application/json" }
    }), n = await e.json();
    if (!e.ok) throw new Error(n.error || "The computer refused the request");
    le.value = "", await Nn();
  } catch (e) {
    dt.textContent = e instanceof Error ? e.message : "Could not send the request";
  } finally {
    Pt.disabled = !1;
  }
}
async function Re() {
  if (Z)
    try {
      const t = await _t("/api/pending"), e = await t.json();
      if (!t.ok) throw new Error(e.error || "Receiver authentication failed");
      const n = $n(e, "pending-v1");
      en.textContent = "";
      for (const r of n)
        lt.has(r.transfer_id) || await Lr(r);
      Ln.classList.toggle("hidden", lt.size > 0), await Nn();
    } catch (t) {
      en.textContent = t instanceof Error ? t.message : "Could not reach Heliox";
    } finally {
      clearTimeout(he), Z && (he = window.setTimeout(Re, 1800));
    }
}
async function Lr(t) {
  if (!Z) throw new Error("This phone is not paired");
  const e = `/api/transfers/${encodeURIComponent(t.transfer_id)}`, n = await _t(e);
  if (!n.ok) {
    const f = await n.json();
    throw new Error(f.error || "Could not download the handoff");
  }
  const r = new Uint8Array(await n.arrayBuffer());
  if (r.length < 29) throw new Error("The encrypted handoff is incomplete");
  const o = Xt(
    W(Z.device_secret),
    r.slice(0, 12),
    V.encode(`transfer-v1:${t.transfer_id}`)
  ).decrypt(r.slice(12)), i = new Blob([new Uint8Array(o)], { type: t.mime_type }), s = URL.createObjectURL(i), c = t.mime_type === "text/plain" && o.length <= 2e4 ? ve.decode(o) : "";
  Cr(t, s, c), lt.add(t.transfer_id);
}
function Cr(t, e, n = "") {
  const r = document.createElement("article");
  if (r.className = "transfer", t.mime_type.startsWith("image/")) {
    const u = document.createElement("img");
    u.src = e, u.alt = t.filename, r.append(u);
  }
  const o = document.createElement("div");
  o.className = "transfer-info";
  const i = document.createElement("strong");
  i.textContent = t.filename;
  const s = document.createElement("span");
  if (s.textContent = `${t.kind} · ${Ur(t.size)}`, n) {
    const u = document.createElement("p");
    u.className = "answer", u.textContent = n, o.append(u);
  }
  const c = document.createElement("div");
  c.className = "transfer-actions";
  const f = document.createElement("a");
  f.href = e, f.download = t.filename, f.textContent = "Save";
  const a = document.createElement("button");
  a.type = "button", a.textContent = "Received", a.addEventListener("click", async () => {
    const u = `/api/transfers/${encodeURIComponent(t.transfer_id)}/ack`;
    (await _t(u, {
      method: "POST",
      body: "{}",
      headers: { "Content-Type": "application/json" }
    })).ok && (r.remove(), URL.revokeObjectURL(e), lt.delete(t.transfer_id), Ln.classList.toggle("hidden", lt.size > 0));
  }), c.append(f, a), o.append(i, s, c), r.append(o), Cn.prepend(r);
}
function Ur(t) {
  return t < 1024 ? `${t} B` : t < 1024 * 1024 ? `${(t / 1024).toFixed(1)} KB` : `${(t / (1024 * 1024)).toFixed(1)} MB`;
}
function Oe() {
  const t = !!Z;
  Er.classList.toggle("hidden", t), xr.classList.toggle("hidden", !t), t || Un.classList.add("hidden"), nn.classList.toggle("online", t), nn.textContent = t ? "Paired · encrypted" : "Not paired", clearTimeout(he), t && Re();
}
yt.addEventListener("click", () => void Ar());
Pt.addEventListener("click", () => void Ir());
Br.addEventListener("click", _r);
Oe();
