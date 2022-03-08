///////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (C) 2007-2022 Cedric Guillemet
//
//    Immath is free software: you can redistribute it and/or modify
//    it under the terms of the GNU General Public License as published by
//    the Free Software Foundation, either version 3 of the License, or
//    (at your option) any later version.
//
//    Immath is distributed in the hope that it will be useful,
//    but WITHOUT ANY WARRANTY; without even the implied warranty of
//    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//    GNU General Public License for more details.
//
//    You should have received a copy of the GNU General Public License
//    along with Immath.  If not, see <http://www.gnu.org/licenses/>
//
///////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <memory.h>
#include <float.h>

namespace Imm
{

struct matrix;

inline void FPU_MatrixF_x_MatrixF(const float *a, const float *b, float *r)
{
	r[0] = a[0]*b[0] + a[1]*b[4] + a[2]*b[8]  + a[3]*b[12];
	r[1] = a[0]*b[1] + a[1]*b[5] + a[2]*b[9]  + a[3]*b[13];
	r[2] = a[0]*b[2] + a[1]*b[6] + a[2]*b[10] + a[3]*b[14];
	r[3] = a[0]*b[3] + a[1]*b[7] + a[2]*b[11] + a[3]*b[15];

	r[4] = a[4]*b[0] + a[5]*b[4] + a[6]*b[8]  + a[7]*b[12];
	r[5] = a[4]*b[1] + a[5]*b[5] + a[6]*b[9]  + a[7]*b[13];
	r[6] = a[4]*b[2] + a[5]*b[6] + a[6]*b[10] + a[7]*b[14];
	r[7] = a[4]*b[3] + a[5]*b[7] + a[6]*b[11] + a[7]*b[15];

	r[8] = a[8]*b[0] + a[9]*b[4] + a[10]*b[8] + a[11]*b[12];
	r[9] = a[8]*b[1] + a[9]*b[5] + a[10]*b[9] + a[11]*b[13];
	r[10]= a[8]*b[2] + a[9]*b[6] + a[10]*b[10]+ a[11]*b[14];
	r[11]= a[8]*b[3] + a[9]*b[7] + a[10]*b[11]+ a[11]*b[15];

	r[12]= a[12]*b[0]+ a[13]*b[4]+ a[14]*b[8] + a[15]*b[12];
	r[13]= a[12]*b[1]+ a[13]*b[5]+ a[14]*b[9] + a[15]*b[13];
	r[14]= a[12]*b[2]+ a[13]*b[6]+ a[14]*b[10]+ a[15]*b[14];
	r[15]= a[12]*b[3]+ a[13]*b[7]+ a[14]*b[11]+ a[15]*b[15];
}

const float PI    =  3.14159265358979323846f;
const float RAD2DEG = (180.f / PI);
const float DEG2RAD = (PI / 180.f);

template<typename T> T Clamp(T v, T l, T h)
{
    return (v < l) ? l : ((v > h) ? h : v);
}

template<typename T> T Lerp(T a, T b, float t)
{
    return a + (b - a) * t;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

struct vec4
{
public:
	vec4(const vec4& other) : x(other.x), y(other.y), z(other.z), w(other.w) {}
	vec4() {}
	vec4(float _x, float _y, float _z = 0.f, float _w = 0.f) : x(_x), y(_y), z(_z), w(_w)
	{
	}
	vec4(int _x, int _y, int _z = 0, int _w = 0) : x((float)_x), y((float)_y), z((float)_z), w((float)_w)
	{
	}
	
	vec4 (uint32_t col) { fromUInt32(col); }
	vec4 (float v ) : x(v), y(v), z(v), w(v) {}

	float x,y,z,w;

	void Lerp(const vec4& v, float t)
	{
		x += (v.x-x) * t;
		y += (v.y-y) * t;
		z += (v.z-z) * t;
		w += (v.w-w) * t;
	}
	void LerpColor(const vec4& v, float t)
	{
		for (int i = 0; i < 4;i++)
			(*this)[i] = sqrtf(((*this)[i] * (*this)[i]) * (1.f - t) + (v[i] * v[i]) * (t));
	}
    void Lerp(const vec4& v, const vec4& v2,float t)
	{
        *this = v;
        Lerp(v2, t);
	}
    
	inline void set(float v) { x = y = z = w = v; }
	inline void set(float _x, float _y, float _z = 0.f, float _w = 0.f)	{ x = _x; y = _y; z = _z; w = _w; }

	inline vec4& operator -= ( const vec4& v ) { x -= v.x; y -= v.y; z -= v.z; w -= v.w; return *this; }
	inline vec4& operator += ( const vec4& v ) { x += v.x; y += v.y; z += v.z; w += v.w; return *this; }
	inline vec4& operator *= ( const vec4& v ) { x *= v.x; y *= v.y; z *= v.z; w *= v.w; return *this; }
	inline vec4& operator *= ( float v ) { x *= v;	y *= v;	z *= v;	w *= v;	return *this; }

	inline vec4 operator * ( float f ) const;
	inline vec4 operator - () const;
	inline vec4 operator - ( const vec4& v ) const;
	inline vec4 operator + ( const vec4& v ) const;
	inline vec4 operator * ( const vec4& v ) const;
	
	inline const vec4& operator + () const { return (*this); }
	inline float length() const { return sqrtf(x*x +y*y +z*z ); };
	inline float lengthSq() const { return (x*x +y*y +z*z ); };
	inline vec4 normalize() { (*this) *= (1.f/length()+FLT_EPSILON); return (*this); }
	inline vec4 normalize(const vec4& v) { this->set(v.x, v.y, v.z, v.w); this->normalize(); return (*this); }
	inline int LongestAxis() const 
	{
		int res = 0; 
		res = (fabsf((*this)[1]) > fabsf((*this)[res])) ? 1 : res;
		res = (fabsf((*this)[2]) > fabsf((*this)[res])) ? 2 : res;
		return res;
	}
	inline void cross(const vec4& v)
	{
		vec4 res;
		res.x = y * v.z - z * v.y;
		res.y = z * v.x - x * v.z;
		res.z = x * v.y - y * v.x;

		x = res.x;
		y = res.y;
		z = res.z;
		w = 0.f;
	}
	inline void cross(const vec4& v1, const vec4& v2)
	{
		x = v1.y * v2.z - v1.z * v2.y;
		y = v1.z * v2.x - v1.x * v2.z;
		z = v1.x * v2.y - v1.y * v2.x;
		w = 0.f;
	}
	inline float dot( const vec4 &v) const
	{
		return (x * v.x) + (y * v.y) + (z * v.z) + (w * v.w);
	}

	void isMaxOf(const vec4& v)
	{
		x = (v.x>x)?v.x:x;
		y = (v.y>y)?v.y:y;
		z = (v.z>z)?v.z:z;
		w = (v.w>w)?v.z:w;
	}
	void isMinOf(const vec4& v)
	{
		x = (v.x>x)?x:v.x;
		y = (v.y>y)?y:v.y;
		z = (v.z>z)?z:v.z;
		w = (v.w>w)?z:v.w;
	}

	bool isInside( const vec4& min, const vec4& max ) const
	{
		if ( min.x > x || max.x < x ||
			min.y > y || max.y < y ||
			min.z > z || max.z < z  )
			return false;
		return true;
	}
    
    vec4 symetrical(const vec4& v) const
    {
        vec4 res;
        float dist = signedDistanceTo(v);
        res = v;
        res -= (*this)*dist*2.f;
        
        return res;
    }
	void transform(const matrix& matrix );
	void transform(const vec4 & s, const matrix& matrix );

	void TransformVector(const matrix& matrix );
	void TransformPoint(const matrix& matrix );
	void TransformVector(const vec4& v, const matrix& matrix ) { (*this) = v; this->TransformVector(matrix); }
	void TransformPoint(const vec4& v, const matrix& matrix ) { (*this) = v; this->TransformPoint(matrix); }

    // quaternion slerp
    //void slerp(const vec4 &q1, const vec4 &q2, float t );

	inline float signedDistanceTo(const vec4& point) const;
	vec4 interpolateHermite(const vec4 &nextKey, const vec4 &nextKeyP1, const vec4 &prevKey, float ratio) const;
    static float d(const vec4& v1, const vec4& v2) { return (v1-v2).length(); }
    static float d2(const vec4& v1, const vec4& v2) { return (v1-v2).lengthSq(); }
    
	static vec4 zero;

    uint16_t toUInt5551() const { return (uint16_t)(((int)(w*1.f)<< 15) + ((int)(z*31.f)<< 10) + ((int)(y*31.f)<< 5) + ((int)(x*31.f))); }
    void fromUInt5551(unsigned short v) { w = (float)( (v&0x8000) >> 15) ; z = (float)( (v&0x7C00) >> 10) * (1.f/31.f); 
	y = (float)( (v&0x3E0) >> 5) * (1.f/31.f); x = (float)( (v&0x1F)) * (1.f/31.f); } 

	uint32_t toUInt32() const { return ((int)(w*255.f)<< 24) + ((int)(z*255.f)<< 16) + ((int)(y*255.f)<< 8) + ((int)(x*255.f)); }
	void fromUInt32(uint32_t v) { w = (float)( (v&0xFF000000) >> 24) * (1.f/255.f); z = (float)( (v&0xFF0000) >> 16) * (1.f/255.f);
	y = (float)( (v&0xFF00) >> 8) * (1.f/255.f); x = (float)( (v&0xFF)) * (1.f/255.f); } 

    vec4 swapedRB() const;
	float& operator [] (size_t index) { return ((float*)&x)[index]; }
	const float& operator [] (size_t index) const { return ((float*)&x)[index]; }
};

inline vec4 vec4::operator * ( float f ) const { return vec4(x * f, y * f, z * f, w *f); }
inline vec4 vec4::operator - () const { return vec4(-x, -y, -z, -w); }
inline vec4 vec4::operator - ( const vec4& v ) const { return vec4(x - v.x, y - v.y, z - v.z, w - v.w); }
inline vec4 vec4::operator + ( const vec4& v ) const { return vec4(x + v.x, y + v.y, z + v.z, w + v.w); }
inline vec4 vec4::operator * ( const vec4& v ) const { return vec4(x * v.x, y * v.y, z * v.z, w * v.w); }
inline float vec4::signedDistanceTo(const vec4& point) const	{ return (point.dot(vec4(x,y,z))) - w; }

inline vec4 normalized(const vec4& v) { vec4 res; res = v; res.normalize(); return res; }
inline vec4 cross(const vec4& v1, const vec4& v2)
{
    vec4 res;
    res.x = v1.y * v2.z - v1.z * v2.y;
    res.y = v1.z * v2.x - v1.x * v2.z;
    res.z = v1.x * v2.y - v1.y * v2.x;
    res.w = 0.f;
    return res;
}

inline float Dot( const vec4 &v1, const vec4 &v2)
{
	return (v1.x * v2.x) + (v1.y * v2.y) + (v1.z * v2.z);
}

inline float Distance(const vec4& v1, const vec4& v2) { return vec4::d(v1, v2); }
inline float DistanceXY(const vec4& v1, const vec4& v2) { return vec4::d(vec4(v1.x, v1.y), vec4(v2.x, v2.y)); }
inline float DistanceSq(const vec4& v1, const vec4& v2) { return vec4::d2(v1, v2); }

inline vec4 MakeNormal(const vec4 & point1, const vec4 & point2, const vec4 & point3)
{
	vec4 nrm;
	vec4 tmp1 = point1 - point3;
	vec4 tmp2 = point2 - point3;
	nrm.cross(tmp1, tmp2);
	return nrm;
}

inline float vecByIndex(const vec4& v, int idx)
{
	switch( idx)
	{
	case 0: return v.x;
	case 1: return v.y;
	case 2: return v.z;
	default: return v.w;
	}
}

inline vec4 vecMul(const vec4& v1, const vec4& v2)
{
	return vec4(v1.x * v2.x, v1.y * v2.y, v1.z * v2.z , v1.w * v2.w);
}

inline vec4 vecMin(const vec4& v1, const vec4& v2)
{
	vec4 res = v1;
	res.isMinOf(v2);

	return res;
}
inline vec4 vecMax(const vec4& v1, const vec4& v2)
{
	vec4 res = v1;
	res.isMaxOf(v2);
	return res;
}

inline vec4 vecFloor(const vec4& v)
{
	return vec4(floorf(v.x), floorf(v.y), floorf(v.z), floorf(v.w) );
}

inline vec4 splatZ(const vec4& v) { return vec4(v.z); }
inline vec4 splatW(const vec4& v) { return vec4(v.w); }

inline vec4 vecReciprocal(const vec4& v) { return vec4(1.f/v.x, 1.f/v.y, 1.f/v.z, 1.f/v.w); }

inline vec4 buildPlan(const vec4 & p_point1, const vec4 & p_normal)
{
	vec4 normal, res;
	normal.normalize(p_normal);
	res.w = normal.dot(p_point1);
	res.x = normal.x;
	res.y = normal.y;
	res.z = normal.z;
	
	return res;
}
inline vec4 vec4::swapedRB() const { return vec4(z,y,x,w); }

inline float smootherstep(float edge0, float edge1, float x)
{
    // Scale, and clamp x to 0..1 range
    x = Clamp((x - edge0)/(edge1 - edge0), 0.f, 1.f);
    // Evaluate polynomial
    return x*x*x*(x*(x*6 - 15) + 10);
}

inline vec4* slerp(vec4 *pout, const vec4* pq1, const vec4* pq2, float t)
{
    float dot, epsilon;

    epsilon = 1.0f;
    dot = pq1->dot( *pq2 );
    if ( dot < 0.0f ) epsilon = -1.0f;
    pout->x = (1.0f - t) * pq1->x + epsilon * t * pq2->x;
    pout->y = (1.0f - t) * pq1->y + epsilon * t * pq2->y;
    pout->z = (1.0f - t) * pq1->z + epsilon * t * pq2->z;
    pout->w = (1.0f - t) * pq1->w + epsilon * t * pq2->w;
    return pout;
}

inline vec4 TransformPoint(const vec4& v, const matrix& matrix) { vec4 p(v); p.TransformPoint(matrix); return p; }
inline vec4 TransformVector(const vec4& v, const matrix& matrix) { vec4 p(v); p.TransformVector(matrix); return p; }

///////////////////////////////////////////////////////////////////////////////////////////////////

typedef struct vec3
{
    float x,y,z;
    void set(float v) { x = y = z = v; }
    float length() const { return sqrtf( x * x + y * y + z * z ); }
    void lerp( float v, float t)
    {
        x = Lerp( x, v, t);
        y = Lerp( y, v, t);
        z = Lerp( z, v, t);
    }
    vec3& operator = (const vec4& v)
    {
        x = v.x;
        y = v.y;
        z = v.z;
        return *this;
    }
    vec3& operator * (const float v)
    {
        vec3 ret;
        ret.x = x * v;
        ret.y = y * v;
        ret.z = z * v;
        return *this;
    }
    vec3& operator += (const vec4& v)
    {
        x += v.x;
        y += v.y;
        z += v.z;
        return *this;
    }
    vec3& operator += (const vec3& v)
    {
        x += v.x;
        y += v.y;
        z += v.z;
        return *this;
    }
    vec3& operator -= (const vec3& v)
    {
        x -= v.x;
        y -= v.y;
        z -= v.z;
        return *this;
    }
    
    vec4 getVec4() const
    {
        return vec4( x, y, z, 0.f );
    }
} vec3;

///////////////////////////////////////////////////////////////////////////////////////////////////

struct matrix
{
public:
	union
	{
		float m[4][4];
		float m16[16];
		struct 
		{
            vec4 right, up, dir, position;
		};
	};

	matrix(float v1, float v2, float v3, float v4, float v5, float v6, float v7, float v8, float v9, float v10, float v11, float v12, float v13, float v14, float v15, float v16)
    {
		m16[0] = v1;
		m16[1] = v2;
		m16[2] = v3;
		m16[3] = v4;
		m16[4] = v5;
		m16[5] = v6;
		m16[6] = v7;
		m16[7] = v8;
		m16[8] = v9;
		m16[9] = v10;
		m16[10] = v11;
		m16[11] = v12;
		m16[12] = v13;
		m16[13] = v14;
		m16[14] = v15;
		m16[15] = v16;
    }
	matrix(const matrix& other) { memcpy(&m16[0], &other.m16[0], sizeof(float) * 16); }
	matrix(const vec4 & r, const vec4 &u, const vec4& d, const vec4& p) { set(r, u, d, p); }
    matrix() {}
	void set(const vec4 & r, const vec4 &u, const vec4& d, const vec4& p) { right=r; up=u; dir=d; position=p; }
	void set(float v1, float v2, float v3, float v4, float v5, float v6, float v7, float v8, float v9, float v10, float v11, float v12, float v13, float v14, float v15, float v16)
	{
		m16[0] = v1;
		m16[1] = v2;
		m16[2] = v3;
		m16[3] = v4;
		m16[4] = v5;
		m16[5] = v6;
		m16[6] = v7;
		m16[7] = v8;
		m16[8] = v9;
		m16[9] = v10;
		m16[10] = v11;
		m16[11] = v12;
		m16[12] = v13;
		m16[13] = v14;
		m16[14] = v15;
		m16[15] = v16;
	}
	static matrix GetIdentity() { return matrix(1.f, 0.f, 0.f, 0.f,
		0.f, 1.f, 0.f, 0.f,
		0.f, 0.f, 1.f, 0.f,
		0.f, 0.f, 0.f, 1.f);
	}
	operator float * () { return m16; }
	operator const float* () const { return m16; }
	void translation(float _x, float _y, float _z) { this->translation( vec4(_x, _y, _z) ); }
	
	void translation(const vec4& vt)
	{ 
		right.set(1.f, 0.f, 0.f, 0.f); 
		up.set(0.f, 1.f, 0.f, 0.f); 
		dir.set(0.f, 0.f, 1.f, 0.f); 
		position.set(vt.x, vt.y, vt.z, 1.f); 
	}
	void translationScale(const vec4& vt, const vec4& scale)
	{
		right.set(scale.x, 0.f, 0.f, 0.f);
		up.set(0.f, scale.y, 0.f, 0.f);
		dir.set(0.f, 0.f, scale.z, 0.f);
		position.set(vt.x, vt.y, vt.z, 1.f);
	}

	inline void rotationY(const float angle )
	{
		float c = cosf(angle);
		float s = sinf(angle);

		right.set(c, 0.f, -s, 0.f);
		up.set(0.f, 1.f, 0.f , 0.f);
		dir.set(s, 0.f, c , 0.f);
		position.set(0.f, 0.f, 0.f , 1.f);
	}

	inline void rotationX(const float angle )
	{
		float c = cosf(angle);
		float s = sinf(angle);

		right.set(1.f, 0.f , 0.f, 0.f);
		up.set(0.f, c , s, 0.f);
		dir.set(0.f, -s, c, 0.f);
		position.set(0.f, 0.f , 0.f, 1.f);
	}

	inline void rotationZ(const float angle )
	{
		float c = cosf(angle);
		float s = sinf(angle);

		right.set(c , s, 0.f, 0.f);
		up.set(-s, c, 0.f, 0.f);
		dir.set(0.f , 0.f, 1.f, 0.f);
		position.set(0.f , 0.f, 0, 1.f);
	}
	inline void scale(float _s)
	{
		right.set(_s, 0.f, 0.f, 0.f); 
		up.set(0.f, _s, 0.f, 0.f); 
		dir.set(0.f, 0.f, _s, 0.f); 
		position.set(0.f, 0.f, 0.f, 1.f); 
	}
	inline void scale(float _x, float _y, float _z)
	{
		right.set(_x, 0.f, 0.f, 0.f); 
		up.set(0.f, _y, 0.f, 0.f); 
		dir.set(0.f, 0.f, _z, 0.f); 
		position.set(0.f, 0.f, 0.f, 1.f); 
	}
	inline void scale(const vec4& s) { scale(s.x, s.y, s.z); }

	inline matrix& operator *= ( const matrix& mat )
	{
		matrix tmpMat;
		tmpMat = *this;
		tmpMat.Multiply(mat);
		*this = tmpMat;
		return *this;
	}
	inline matrix operator * (const matrix& mat) const
	{
		matrix matT;
		matT.Multiply(*this, mat);
		return matT;
	}

	inline void Multiply( const matrix& mat)
	{
		matrix tmp;
		tmp = *this;

		FPU_MatrixF_x_MatrixF( (float*)&tmp, (float*)&mat, (float*)this);
	}

	inline void Multiply( const matrix &m1, const matrix &m2 )
	{
		FPU_MatrixF_x_MatrixF( (float*)&m1, (float*)&m2, (float*)this);
	}

	void glhPerspectivef2(float fovyInDegrees, float aspectRatio, float znear, float zfar, bool homogeneousNdc, bool rightHand = false);
	void glhPerspectivef2Rad(float fovyRad, float aspectRatio, float znear, float zfar, bool homogeneousNdc, bool rightHand = false);
	
	void glhFrustumf2(float x, float y, float width, float height,	float znear, float zfar, bool homogeneousNdc, bool rightHand = false);
	void PerspectiveFovLH2(const float fovy, const float aspect, const float zn, const float zf );
	void OrthoOffCenterLH(const float l, float r, float b, const float t, float zn, const float zf );
	void lookAtRH(const vec4 &eye, const vec4 &at, const vec4 &up );
	void lookAtLH(const vec4 &eye, const vec4 &at, const vec4 &up );
	void LookAt(const vec4 &eye, const vec4 &at, const vec4 &up );
    void rotationQuaternion( const vec4 &q );

	inline float GetDeterminant() const
	{
		return m[0][0] * m[1][1] * m[2][2] + m[0][1] * m[1][2] * m[2][0] +    m[0][2] * m[1][0] * m[2][1] -
			m[0][2] * m[1][1] * m[2][0] - m[0][1] * m[1][0] * m[2][2] -    m[0][0] * m[1][2] * m[2][1];
	}

	float inverse(const matrix &srcMatrix, bool affine = false );
	float inverse(bool affine=false);
	void identity() { 		
		right.set(1.f, 0.f, 0.f, 0.f); 
		up.set(0.f, 1.f, 0.f, 0.f); 
		dir.set(0.f, 0.f, 1.f, 0.f); 
		position.set(0.f, 0.f, 0.f, 1.f); 
	}
	inline void transpose()
	{
		matrix tmpm;
		for (int l = 0; l < 4; l++)
		{
			for (int c = 0; c < 4; c++)
			{
				tmpm.m[l][c] = m[c][l];
			}
		}
		(*this) = tmpm;
	}
	void rotationAxis(const vec4 & axis, float angle );
	void lerp(const matrix& r, const matrix& t, float s)
	{ 
		right = Lerp(r.right, t.right, s);
		up = Lerp(r.up, t.up, s);
		dir = Lerp(r.dir, t.dir, s);
		position = Lerp(r.position, t.position, s);
	}
	void rotationYawPitchRoll(const float yaw, const float pitch, const float roll );

	inline void orthoNormalize()
	{
		right.normalize();
		up.normalize();
		dir.normalize();
	}

    static matrix Identity;


};

#ifdef MACOSX
#include <xmmintrin.h> //declares _mm_* intrinsics
#endif
//#include <intrin.h>
/*
#ifdef WIN32
inline int FloatToInt_SSE(float x)
{
    return _mm_cvt_ss2si( _mm_load_ss(&x) );
}
#endif
*/
extern int g_seed;
inline int fastrand() 
{ 
	g_seed = (214013*g_seed+2531011); 
	return (g_seed>>16)&0x7FFF; 
} 

inline float r01()
{
	return ((float)fastrand())*(1.f/32767.f);
}



inline bool CollisionClosestPointOnSegment( const vec4 & point, const vec4 & vertPos1, const vec4 & vertPos2, vec4& res )
{
    
    vec4 c = point - vertPos1;
    vec4 V;
    
    V.normalize(vertPos2 - vertPos1);
    float d = (vertPos2 - vertPos1).length();
    float t = V.dot(c);
    
    if (t < 0.f)
    {
        return false;//vertPos1;
    }
    
    if (t > d)
    {
        return false;//vertPos2;
    }
    
    res = vertPos1 + V * t;
    return true;
}

inline vec4 CollisionClosestPointOnSegment( const vec4 & point, const vec4 & vertPos1, const vec4 & vertPos2 )
{
    
    vec4 c = point - vertPos1;
    vec4 V;
    
    V.normalize(vertPos2 - vertPos1);
    float d = (vertPos2 - vertPos1).length();
    float t = V.dot(c);
    
    if (t < 0.f)
    {
        return vertPos1;
    }
    
    if (t > d)
    {
        return vertPos2;
    }
    
    return vertPos1 + V * t;
}

inline float DistanceCollisionClosestPointOnSegment( const vec4 & point, const vec4 & vertPos1, const vec4 & vertPos2 )
{
    vec4 c = point - vertPos1;
    vec4 V;
    
	if ((vertPos2 - vertPos1).lengthSq() < 0.001f)
		return Distance(point, vertPos1);

    V.normalize(vertPos2 - vertPos1);
    float d = (vertPos2 - vertPos1).length();
    float t = V.dot(c);
    
    if (t < 0.f)
    {
        return (vertPos1-point).length();
    }
    
    if (t > d)
    {
        return (vertPos2-point).length();
    }
    
    vec4 r = vertPos1 + V * t;
    return ( r - point ).length();
}

///////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T> struct PID
{
        inline PID()
        {
                Ki = 1;
                Kp = 1;
                Kd = 1;
                error = previouserror = I = 0.f;
        }
        inline PID(float _Ki, float _Kp, float _Kd)
        {
                Ki = _Ki;
                Kp = _Kp;
                Kd = _Kd;
                error = previouserror = I = 0.f;
        }
        inline void SetIPD(float _Ki, float _Kp, float _Kd)
        {
                Ki = _Ki;
                Kp = _Kp;
                Kd = _Kd;
        }
        /*
        start:
        previous_error = error or 0 if undefined
        error = setpoint - actual_position
        P = Kp * error
        I = I + Ki * error * dt
        D = Kd * (error - previous_error) / dt
        output = P + I + D
        wait(dt)
        goto start
        */
        inline T Compute(const T& desiredPos, const T& currentPos, float dt)
        {
                previouserror = error;
                error = desiredPos - currentPos;
                T P = Kp * error;
                I += Ki * error * dt;
                T D = Kd * (error - previouserror) / dt;
                T output = P + I + D;
                return output;
        }

        T error, previouserror, I;
        float Ki, Kp, Kd;
};

typedef PID<float> PIDf;

#pragma pack(push)
#pragma pack(1)
struct fixed816_t
{
    char intValue;
    short floatValue;

    float toFloat()
    {
        return static_cast<float>(intValue) + static_cast<float>(floatValue)/32767.f;
    }

    fixed816_t( float v)
    {
        intValue = static_cast<char>(v);
        floatValue = static_cast<unsigned short>(fmodf(v, 1.f) * 32767.f);
    }
};

#pragma pack(pop)

struct ZFrustum  
{
	ZFrustum()
	{
	}
	
    void    Update(const matrix &view, const matrix& projection);
	void	Update(const float* clip);
    bool    PointInFrustum( const vec4 & vt ) const
    {
        // If you remember the plane equation (A*x + B*y + C*z + D = 0), then the rest
        // of this code should be quite obvious and easy to figure out yourself.
        // In case don't know the plane equation, it might be a good idea to look
        // at our Plane Collision tutorial at www.GameTutorials.com in OpenGL Tutorials.
        // I will briefly go over it here.  (A,B,C) is the (X,Y,Z) of the normal to the plane.
        // They are the same thing... but just called ABC because you don't want to say:
        // (x*x + y*y + z*z + d = 0).  That would be wrong, so they substitute them.
        // the (x, y, z) in the equation is the point that you are testing.  The D is
        // The Distance the plane is from the origin.  The equation ends with "= 0" because
        // that is true when the point (x, y, z) is ON the plane.  When the point is NOT on
        // the plane, it is either a negative number (the point is behind the plane) or a
        // positive number (the point is in front of the plane).  We want to check if the point
        // is in front of the plane, so all we have to do is go through each point and make
        // sure the plane equation goes out to a positive number on each side of the frustum.
        // The result (be it positive or negative) is the Distance the point is front the plane.

        // Go through all the sides of the frustum
        for(int i = 0; i < 6; i++ )
        {
            // Calculate the plane equation and check if the point is behind a side of the frustum
            if(m_Frustum[i][A] * vt.x + m_Frustum[i][B] * vt.y + m_Frustum[i][C] * vt.z + m_Frustum[i][D] <= 0)
            {
                // The point was behind a side, so it ISN'T in the frustum
                return false;
            }
        }

        // The point was inside of the frustum (In front of ALL the sides of the frustum)
        return true;
    }
    
    bool    SphereInFrustum( const vec4 & vt) const
    {
        for(int i = 0; i < 6; i++ )    
        {
            // If the center of the sphere is farther away from the plane than the radius
            if( m_Frustum[i][A] * vt.x + m_Frustum[i][B] * vt.y + m_Frustum[i][C] * vt.z + m_Frustum[i][D] <= -vt.w )
            {
                // The Distance was greater than the radius so the sphere is outside of the frustum
                return false;
            }
        }
        
        // The sphere was inside of the frustum!
        return true;
    }

	int SphereInFrustumVis(const vec4& v) const
	{

		float Distance;
		int result = 2;

		for(int i=0; i < 6; i++) {
			Distance = m_Frustum[i][A] * v.x + m_Frustum[i][B] * v.y + m_Frustum[i][C] * v.z + m_Frustum[i][D];//pl[i].Distance(p);
			if (Distance < -v.w)
				return 0;
			else if (Distance < v.w)
				result =  1;
		}
		return(result);
	}

    bool    BoxInFrustum( const vec4 & vt, const vec4 & size ) const
    {
        for(int i = 0; i < 6; i++ )
        {
            if(m_Frustum[i][A] * (vt.x - size.x) + m_Frustum[i][B] * (vt.y - size.y) + m_Frustum[i][C] * (vt.z - size.z) + m_Frustum[i][D] > 0)
            continue;
            if(m_Frustum[i][A] * (vt.x + size.x) + m_Frustum[i][B] * (vt.y - size.y) + m_Frustum[i][C] * (vt.z - size.z) + m_Frustum[i][D] > 0)
            continue;
            if(m_Frustum[i][A] * (vt.x - size.x) + m_Frustum[i][B] * (vt.y + size.y) + m_Frustum[i][C] * (vt.z - size.z) + m_Frustum[i][D] > 0)
            continue;
            if(m_Frustum[i][A] * (vt.x + size.x) + m_Frustum[i][B] * (vt.y + size.y) + m_Frustum[i][C] * (vt.z - size.z) + m_Frustum[i][D] > 0)
            continue;
            if(m_Frustum[i][A] * (vt.x - size.x) + m_Frustum[i][B] * (vt.y - size.y) + m_Frustum[i][C] * (vt.z + size.z) + m_Frustum[i][D] > 0)
            continue;
            if(m_Frustum[i][A] * (vt.x + size.x) + m_Frustum[i][B] * (vt.y - size.y) + m_Frustum[i][C] * (vt.z + size.z) + m_Frustum[i][D] > 0)
            continue;
            if(m_Frustum[i][A] * (vt.x - size.x) + m_Frustum[i][B] * (vt.y + size.y) + m_Frustum[i][C] * (vt.z + size.z) + m_Frustum[i][D] > 0)
            continue;
            if(m_Frustum[i][A] * (vt.x + size.x) + m_Frustum[i][B] * (vt.y + size.y) + m_Frustum[i][C] * (vt.z + size.z) + m_Frustum[i][D] > 0)
            continue;

            // If we get here, it isn't in the frustum
            return false;
        }
        return true;
    }

	// matrix is an orthonormalized matrix. only orientation is used.
	bool OBBInFrustum( const matrix &mt, const vec4 &pos, const vec4& size) const;

public:

    float m_Frustum[6][4];
    void NormalizePlane(float frustum[6][4], int side);

    enum FrustumSide
    {
        RIGHT    = 0,        // The RIGHT side of the frustum
        LEFT    = 1,        // The LEFT     side of the frustum
        BOTTOM    = 2,        // The BOTTOM side of the frustum
        TOP        = 3,        // The TOP side of the frustum
        BACK    = 4,        // The BACK    side of the frustum
        FRONT    = 5            // The FRONT side of the frustum
    }; 

    // Like above, instead of saying a number for the ABC and D of the plane, we
    // want to be more descriptive.
    enum PlaneData
    {
        A = 0,                // The X value of the plane's normal
        B = 1,                // The Y value of the plane's normal
        C = 2,                // The Z value of the plane's normal
        D = 3                // The Distance the plane is from the origin
    };
};

///////////////////////////////////////////////////////////////////////////////////////////////////
/*
void CovarianceMatrix(Matrix33 &cov, Point pt[], int numPts)
{
float oon = 1.0f / (float)numPts;
Point c = Point(0.0f, 0.0f, 0.0f);
float e00, e11, e22, e01, e02, e12;
// Compute the center of mass (centroid) of the points
for (int i = 0; i < numPts; i++)
c += pt[i];
c *= oon;
// Compute covariance elements
e00 = e11 = e22 = e01 = e02 = e12 = 0.0f;
for (int i = 0; i < numPts; i++) {
// Translate points so center of mass is at origin
Point p = pt[i] - c;
// Compute covariance of translated points
e00 += p.x * p.x;
e11 += p.y * p.y;
e22 += p.z * p.z;
e01 += p.x * p.y;
e02 += p.x * p.z;
e12 += p.y * p.z;
}
// Fill in the covariance matrix elements
cov[0][0] = e00 * oon;
cov[1][1] = e11 * oon;
cov[2][2] = e22 * oon;
cov[0][1] = cov[1][0] = e01 * oon;
cov[0][2] = cov[2][0] = e02 * oon;
cov[1][2] = cov[2][1] = e12 * oon;
}
*/

inline int NumberOfSetBits(unsigned char i)
{
    i = i - ((i >> 1) & 0x55);
    i = (i & 0x33) + ((i >> 2) & 0x33);
    return (i + (i >> 4)) & 0x0F;
}

struct Segment
{
	Segment() {}
	Segment(vec4 p0, vec4 p1) : P0(p0), P1(p1)
	{
	}

	vec4 P0,P1;
};
struct CapsuleHitHandler;
struct Capsule : public Segment
{
	Capsule() :mask(0),mWeaponIndex(0),mHitStrength(0),mHandle(NULL) {}
	Capsule( vec4 p0, vec4 p1, float rad, CapsuleHitHandler *handle ) : Segment(p0, p1), radius( rad ), mask(0), mHandle(handle),mWeaponIndex(0),mHitStrength(0)
	{
		previousP0 = p0;
		previousP1 = p1;
	}
	float radius;
	vec4 previousP0;
	vec4 previousP1;
	CapsuleHitHandler *mHandle;
	unsigned char mWeaponIndex;
	unsigned char mHitStrength;
	union
	{
		uint32_t mask;
		struct
		{
			uint32_t mBody:1;
			uint32_t mSprawler:1;
			uint32_t mHit:1;
			uint32_t mDull:1;
		};
	};
};

// dist3D_Segment_to_Segment(): get the 3D minimum Distance between 2 segments
//    Input:  two 3D line segments S1 and S2
//    Return: the shortest Distance between S1 and S2
inline float dist3D_Segment_to_Segment(Segment S1, Segment S2)
{
    vec4   u = S1.P1 - S1.P0;
    vec4   v = S2.P1 - S2.P0;
    vec4   w = S1.P0 - S2.P0;
    float    a = Dot(u,u);         // always >= 0
    float    b = Dot(u,v);
    float    c = Dot(v,v);         // always >= 0
    float    d = Dot(u,w);
    float    e = Dot(v,w);
    float    D = a*c - b*b;        // always >= 0
    float    sc, sN, sD = D;       // sc = sN / sD, default sD = D >= 0
    float    tc, tN, tD = D;       // tc = tN / tD, default tD = D >= 0

    // compute the line parameters of the two closest points
    if (D < FLT_EPSILON) { // the lines are almost parallel
        sN = 0.0;         // force using point P0 on segment S1
        sD = 1.0;         // to prevent possible division by 0.0 later
        tN = e;
        tD = c;
    }
    else {                 // get the closest points on the infinite lines
        sN = (b*e - c*d);
        tN = (a*e - b*d);
        if (sN < 0.0) {        // sc < 0 => the s=0 edge is visible
            sN = 0.0;
            tN = e;
            tD = c;
        }
        else if (sN > sD) {  // sc > 1  => the s=1 edge is visible
            sN = sD;
            tN = e + b;
            tD = c;
        }
    }

    if (tN < 0.0) {            // tc < 0 => the t=0 edge is visible
        tN = 0.0;
        // recompute sc for this edge
        if (-d < 0.0)
            sN = 0.0;
        else if (-d > a)
            sN = sD;
        else {
            sN = -d;
            sD = a;
        }
    }
    else if (tN > tD) {      // tc > 1  => the t=1 edge is visible
        tN = tD;
        // recompute sc for this edge
        if ((-d + b) < 0.0)
            sN = 0;
        else if ((-d + b) > a)
            sN = sD;
        else {
            sN = (-d +  b);
            sD = a;
        }
    }
    // finally do the division to get sc and tc
    sc = (fabsf(sN) < FLT_EPSILON ? 0.0f : sN / sD);
    tc = (fabsf(tN) < FLT_EPSILON ? 0.0f : tN / tD);

    // get the difference of the two closest points
    vec4   dP = w + (u * sc) - (v * tc);  // =  S1(sc) - S2(tc)

	return dP.length();   // return the closest Distance
}

static const float RopeDamping     = 0.01f;
/*
static const tvector2 GRAVITY(0, -9.81f);
float DENSITY_OFFSET                       = 100.f;
float GAS_K                                = 0.1f;
float VISC0SITY                            = 0.005f;
*/
inline void SolveVerletRope(vec4& position, vec4& positionOld, vec4& velocity, vec4 acceleration, float timeStep)
{
	vec4 t;
	vec4 oldPos = position;
	acceleration *= timeStep*timeStep;
	t = position - positionOld;
	t *= 1.f-RopeDamping;
	t += acceleration;
	position += t;
	positionOld = oldPos;

	// calculate velocity
	// Velocity = (Position - PositionOld) / dt;
	t = position-positionOld;
	velocity = t*(1.f/timeStep);
}

inline void SolveVerletRope(vec4& position, vec4& positionOld, vec4& velocity, vec4 acceleration, float damping, float timeStep)
{
	vec4 t;
	vec4 oldPos = position;
	acceleration *= timeStep*timeStep;
	t = position - positionOld;
	t *= 1.f-damping;
	t += acceleration;
	position += t;
	positionOld = oldPos;

	// calculate velocity
	// Velocity = (Position - PositionOld) / dt;
	t = position-positionOld;
	velocity = t*(1.f/timeStep);
}

inline int SIGNBIT(float v) { return (v<0.f)?1:0; }

inline vec4 perpStark(vec4 u)
{
    vec4 a = vec4( fabsf(u.x), fabsf(u.y), fabsf(u.z), 0.f );
    unsigned int uyx = SIGNBIT(a.x - a.y);
    unsigned int uzx = SIGNBIT(a.x - a.z);
    unsigned int uzy = SIGNBIT(a.y - a.z);

    unsigned int xm = uyx & uzx;
    unsigned int ym = (1^xm) & uzy;
    unsigned int zm = 1^(xm & ym);

    vec4 v = cross( u, vec4( (float)xm, (float)ym, (float)zm ) );
    return v;
}

inline vec4 computeVecFromPlanNormal( vec4 localDir, float angle, float len )
{
	vec4 newDirX = perpStark(localDir);
	vec4 newDirY;
	newDirY.cross( localDir, newDirX );
	newDirY.normalize();
	vec4 newDir = ( newDirX * cosf( angle ) + newDirY * sinf( angle ) ) * len;
	return newDir;
}

inline vec4 computeRandomVecFromPlanNormal( vec4 localDir )
{
	return computeVecFromPlanNormal( localDir, r01() * PI * 2.f, r01() );
}

inline float roundf(float d)
{
  return floorf(d + 0.5f);
}

inline vec4 Reflect(const vec4 &incidentVec, const vec4 &normal)
{
	return incidentVec - normal * Dot(incidentVec, normal) * 2.f;
}

inline vec4 computePlansIntersections( vec4 n1, float d1, vec4 n2, float d2, vec4 n3, float d3 )
{
	float div = Dot(n1,cross(n2,n3));
	vec4 u = -cross(n2,n3)*d1 - cross(n3,n1)*d2 - cross(n1,n2)*d3;
	vec4 r = u * (1.f/div);
	return r;
}
//////////////////////////////////////////////////////////////////////

struct vec2i
{
	vec2i() {}
	vec2i(const vec2i& other) : x(other.x), y(other.y) {}
	vec2i(int _x, int _y) : x(_x), y(_y) {}

	bool operator == (const vec2i& other) { return (x == other.x && y == other.y); }
	bool operator != (const vec2i& other) { return (x != other.x || y != other.y); }
	void operator += (const vec2i& other) { x += other.x; y += other.y; }
	void operator -= (const vec2i& other) { x -= other.x; y -= other.y; }
	bool operator < (const vec2i& other) const
	{
		if (y < other.y)
			return true;
		if (y > other.y)
			return false;
		if (x < other.x)
			return true;
		if (x > other.x)
			return false;
		return false;
	}
	void ManNormalize()
	{
		x = (x>0) ? 1 : ((x<0) ? -1 : 0);
		y = (y>0) ? 1 : ((y<0) ? -1 : 0);
	}
	int ManLength() { return abs(x + y); }
	float length() const { return sqrtf((float)(x*x + y*y)); }

	vec2i Rotate(float angle) const
	{
		float cs = cosf(angle);
		float sn = sinf(angle);
		return vec2i(int(x*cs - y*sn), int(y*cs + x * sn));
	}
	int x, y;
};

inline vec2i operator + (const vec2i& a, const vec2i &b) { return vec2i(a.x + b.x, a.y + b.y); }
inline vec2i operator - (const vec2i& a, const vec2i &b) { return vec2i(a.x - b.x, a.y - b.y); }
inline vec2i operator * (const vec2i& a, int l) { return vec2i(a.x*l, a.y*l); }

inline vec2i min2i(const vec2i a, const vec2i b)
{
	return vec2i(a.x < b.x ? a.x : b.x, a.y < b.y ? a.y : b.y);
}

inline vec2i max2i(const vec2i a, const vec2i b)
{
	return vec2i(a.x > b.x ? a.x : b.x, a.y > b.y ? a.y : b.y);
}

inline vec4 Rotate(const vec4 & v, float a)
{
	float sn = sinf(a);
	float cs = cosf(a);
	return vec4(v.x * cs - v.y * sn, v.x * sn + v.y * cs, 0.f);
}

inline float GetAngle(const vec4& a, const vec4& b)
{
	vec4 crossed = cross(a, b);
	float dt = Dot(a, b);
	return atan2f(crossed.length(), dt) * ((crossed.z >= 0.f) ? 1.f:-1.f);
}

inline float SignOf(float x)
{
	return (x < 0.f) ? -1.f : 1.f;
}
template<typename T> bool SegmentsIntersect(T x1, T x2, T y1, T y2)
{
	// Assumes x1 <= x2 and y1 <= y2; if this assumption is not safe, the code
	// can be changed to have x1 being min(x1, x2) and x2 being max(x1, x2) and
	// similarly for the ys.
	return x2 >= y1 && y2 >= x1;
}
//static const float ZPI = 3.14159265358979323846f;

inline void DecomposeMatrixToComponents(const float* source, float* translation, float* rotation, float* scale)
{
	matrix mat = *(matrix*)source;

	scale[0] = mat.right.length();
	scale[1] = mat.up.length();
	scale[2] = mat.dir.length();

	mat.orthoNormalize();

	rotation[0] = RAD2DEG * atan2f(mat.m[1][2], mat.m[2][2]);
	rotation[1] = RAD2DEG * atan2f(-mat.m[0][2], sqrtf(mat.m[1][2] * mat.m[1][2] + mat.m[2][2] * mat.m[2][2]));
	rotation[2] = RAD2DEG * atan2f(mat.m[0][1], mat.m[0][0]);

	translation[0] = mat.position.x;
	translation[1] = mat.position.y;
	translation[2] = mat.position.z;
}

inline void RecomposeMatrixFromComponents(const float* translation, const float* rotation, const float* scale, float* destination)
{
	matrix& mat = *(matrix*)destination;
	static const vec4 directionUnary[3] = { vec4(1.f, 0.f, 0.f), vec4(0.f, 1.f, 0.f), vec4(0.f, 0.f, 1.f) };
	matrix rot[3];
	for (int i = 0; i < 3; i++)
	{
		rot[i].rotationAxis(directionUnary[i], rotation[i] * DEG2RAD);
	}

	mat = rot[0] * rot[1] * rot[2];

	float validScale[3];
	for (int i = 0; i < 3; i++)
	{
		if (fabsf(scale[i]) < FLT_EPSILON)
		{
			validScale[i] = 0.001f;
		}
		else
		{
			validScale[i] = scale[i];
		}
	}
	mat.right *= validScale[0];
	mat.up *= validScale[1];
	mat.dir *= validScale[2];
	mat.position = vec4(translation[0], translation[1], translation[2], 1.f);
}

// Static arrays
template <typename T, size_t N>
struct Array
{
   T data[N];
   const size_t size() const { return N; }

   const T operator [] (size_t index) const { return data[index]; }
   operator T* () {
      T* p = new T[N];
      memcpy(p, data, sizeof(data));
      return p;
   }
};

template <typename T, typename ... U> Array(T, U...)->Array<T, 1 + sizeof...(U)>;

} // namespace Imm
