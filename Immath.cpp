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

#include "Immath.h"

using namespace Imm;

int g_seed = 0;
vec4 vec4::zero(0.f, 0.f, 0.f, 0.f);

matrix matrix::Identity(1.f, 0.f, 0.f, 0.f,
        0.f, 1.f, 0.f, 0.f,
        0.f, 0.f, 1.f, 0.f,
        0.f, 0.f, 0.f, 1.f);

void vec4::transform(const matrix& matrix)
{
	vec4 out;

	out.x = x * matrix.m[0][0] + y * matrix.m[1][0] + z * matrix.m[2][0] + w * matrix.m[3][0];
	out.y = x * matrix.m[0][1] + y * matrix.m[1][1] + z * matrix.m[2][1] + w * matrix.m[3][1];
	out.z = x * matrix.m[0][2] + y * matrix.m[1][2] + z * matrix.m[2][2] + w * matrix.m[3][2];
	out.w = x * matrix.m[0][3] + y * matrix.m[1][3] + z * matrix.m[2][3] + w * matrix.m[3][3];

	x = out.x;
	y = out.y;
	z = out.z;
	w = out.w;
}

void vec4::transform(const vec4 & s, const matrix& matrix )
{
	*this = s;
    transform( matrix );
}

void vec4::TransformVector(const matrix& matrix )
{
	vec4 out;

	out.x = x * matrix.m[0][0] + y * matrix.m[1][0] + z * matrix.m[2][0] ;
	out.y = x * matrix.m[0][1] + y * matrix.m[1][1] + z * matrix.m[2][1] ;
	out.z = x * matrix.m[0][2] + y * matrix.m[1][2] + z * matrix.m[2][2] ;
	out.w = x * matrix.m[0][3] + y * matrix.m[1][3] + z * matrix.m[2][3] ;

	x = out.x;
	y = out.y;
	z = out.z;
	w = out.w;
}

void vec4::TransformPoint(const matrix& matrix )
{
	vec4 out;

	out.x = x * matrix.m[0][0] + y * matrix.m[1][0] + z * matrix.m[2][0] + matrix.m[3][0] ;
	out.y = x * matrix.m[0][1] + y * matrix.m[1][1] + z * matrix.m[2][1] + matrix.m[3][1] ;
	out.z = x * matrix.m[0][2] + y * matrix.m[1][2] + z * matrix.m[2][2] + matrix.m[3][2] ;
	out.w = x * matrix.m[0][3] + y * matrix.m[1][3] + z * matrix.m[2][3] + matrix.m[3][3] ;

	x = out.x;
	y = out.y;
	z = out.z;
	w = out.w;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
//matrix will receive the calculated perspective matrix.
//You would have to upload to your shader
// or use glLoadMatrixf if you aren't using shaders.
void matrix::glhPerspectivef2(float fovyInDegrees, float aspectRatio,
									  float znear, float zfar, bool homogeneousNdc, bool rightHand)
{
	const float height = 1.0f / tanf(fovyInDegrees * DEG2RAD * 0.5f);
	const float width = height * 1.0f / aspectRatio;
	glhFrustumf2(0.0f, 0.0f, width, height, znear, zfar, homogeneousNdc, rightHand);
}

void matrix::glhPerspectivef2Rad(float fovyRad, float aspectRatio,
	float znear, float zfar, bool homogeneousNdc, bool rightHand)
{
	const float height = 1.0f / tanf(fovyRad * 0.5f);
	const float width = height * 1.0f / aspectRatio;
	glhFrustumf2(0.0f, 0.0f, width, height, znear, zfar, homogeneousNdc, rightHand);
}

///////////////////////////////////////////////////////////////////////////////////////////////////

void matrix::glhFrustumf2(float x, float y, float width, float height,
								  float znear, float zfar, bool homogeneousNdc, bool rightHand)
{
	const float diff = zfar - znear;
	const float aa = homogeneousNdc ? (zfar + znear) / diff : zfar / diff;
	const float bb = homogeneousNdc ? (2.0f * zfar * znear) / diff : znear * aa;

	m16[0] = width;
	m16[1] = 0.f;
	m16[2] = 0.f;
	m16[3] = 0.f;
	m16[4] = 0.f;
	m16[5] = height;
	m16[6] = 0.f;
	m16[7] = 0.f;
	m16[8] = rightHand ? x : -x;
	m16[9] = rightHand ? y : -y;
	m16[10] = rightHand ? -aa : aa;
	m16[11] = rightHand ? -1.0f : 1.0f;
	m16[12] = 0.f;
	m16[13] = 0.f;
	m16[14] = -bb;
	m16[15] = 0.f;
}

void matrix::lookAtRH(const vec4 &eye, const vec4 &at, const vec4 &up )
{	
	vec4 X, Y, Z, tmp;
	
	Z.normalize(eye - at);
	Y.normalize(up);
	
	tmp.cross(Y, Z);
	X.normalize(tmp);
	
	tmp.cross(Z, X);
	Y.normalize(tmp);
	
	m[0][0] = X.x;
	m[0][1] = Y.x;
	m[0][2] = Z.x;
	m[0][3] = 0.0f;
	
	m[1][0] = X.y;
	m[1][1] = Y.y;
	m[1][2] = Z.y;
	m[1][3] = 0.0f;
	
	m[2][0] = X.z;
	m[2][1] = Y.z;
	m[2][2] = Z.z;
	m[2][3] = 0.0f;
	
	m[3][0] = -X.dot(eye);
	m[3][1] = -Y.dot(eye);
	m[3][2] = -Z.dot(eye);
	m[3][3] = 1.0f;
	
}


void matrix::lookAtLH(const vec4 &eye, const vec4 &at, const vec4 &up )
{
	vec4 X, Y, Z, tmp;

	Z.normalize(at - eye);
	Y.normalize(up);

	tmp.cross(Y, Z);
	X.normalize(tmp);

	tmp.cross(Z, X);
	Y.normalize(tmp);

	m[0][0] = X.x;
	m[0][1] = Y.x;
	m[0][2] = Z.x;
	m[0][3] = 0.0f;

	m[1][0] = X.y;
	m[1][1] = Y.y;
	m[1][2] = Z.y;
	m[1][3] = 0.0f;

	m[2][0] = X.z;
	m[2][1] = Y.z;
	m[2][2] = Z.z;
	m[2][3] = 0.0f;

	m[3][0] = -X.dot(eye);
	m[3][1] = -Y.dot(eye);
	m[3][2] = -Z.dot(eye);
	m[3][3] = 1.0f;
}

void matrix::LookAt(const vec4 &eye, const vec4 &at, const vec4 &up )
{

	vec4 X, Y, Z, tmp;

	Z.normalize(eye - at);
	Y.normalize(up);

	tmp.cross(Y, Z);
	X.normalize(tmp);

	tmp.cross(Z, X);
	Y.normalize(tmp);

	m[0][0] = X.x;
	m[0][1] = X.y;
	m[0][2] = X.z;
	m[0][3] = 0.0f;

	m[1][0] = Y.x;
	m[1][1] = Y.y;
	m[1][2] = Y.z;
	m[1][3] = 0.0f;

	m[2][0] = Z.x;
	m[2][1] = Z.y;
	m[2][2] = Z.z;
	m[2][3] = 0.0f;

	m[3][0] = eye.x;
	m[3][1] = eye.y;
	m[3][2] = eye.z;
	m[3][3] = 1.0f;
}

void matrix::PerspectiveFovLH2(const float fovy, const float aspect, const float zn, const float zf )
{
/*
	xScale     0          0               0
0        yScale       0               0
0          0       zf/(zf-zn)         1
0          0       -zn*zf/(zf-zn)     0
where:
*/
/*
+    pout->m[0][0] =3D 1.0f / (aspect * tan(fovy/2.0f));
+    pout->m[1][1] =3D 1.0f / tan(fovy/2.0f);
+    pout->m[2][2] =3D zf / (zf - zn);
+    pout->m[2][3] =3D 1.0f;
+    pout->m[3][2] =3D (zf * zn) / (zn - zf);
+    pout->m[3][3] =3D 0.0f;



float yscale = cosf(fovy*0.5f);

float xscale = yscale / aspect;

*/
	m[0][0] = 1.0f / (aspect * tanf(fovy*0.5f));
	m[0][1] = 0.0f;
	m[0][2] = 0.0f;
	m[0][3] = 0.0f;

	m[1][0] = 0.0f;
	m[1][1] = 1.0f / tanf(fovy*0.5f);
	m[1][2] = 0.0f;
	m[1][3] = 0.0f;

	m[2][0] = 0.0f;
	m[2][1] = 0.0f;
	m[2][2] = zf / (zf - zn);
	m[2][3] = 1.0f;

	m[3][0] = 0.0f;
	m[3][1] = 0.0f;
	m[3][2] = (zf * zn) / (zn - zf);
	m[3][3] = 0.0f;
}



void matrix::OrthoOffCenterLH(const float l, float r, float b, const float t, float zn, const float zf )
{
	m[0][0] = 2 / (r-l);
	m[0][1] = 0.0f;
	m[0][2] = 0.0f;
	m[0][3] = 0.0f;

	m[1][0] = 0.0f;
	m[1][1] = 2 / (t-b);
	m[1][2] = 0.0f;
	m[1][3] = 0.0f;

	m[2][0] = 0.0f;
	m[2][1] = 0.0f;
	m[2][2] = 1.0f / (zf - zn);
	m[2][3] = 0.0f;

	m[3][0] = (l+r)/(l-r);
	m[3][1] = (t+b)/(b-t);
	m[3][2] = zn / (zn - zf);
	m[3][3] = 1.0f;
}


vec4 vec4::interpolateHermite(const vec4 &nextKey, const vec4 &nextKeyP1, const vec4 &prevKey, float ratio) const
{
	//((tvec43*)res)->Lerp(m_Value, nextKey.m_Value, ratio );
	//return *((tvec43*)res);
	float t = ratio;
	float t2 = t*t;
	float t3 = t2*t;
	float h1 = 2.f*t3 - 3.f*t2 + 1.0f;
	float h2 = -2.f*t3 + 3.f*t2;
	float h3 = (t3 - 2.f*t2 + t) * .5f;
	float h4 = (t3 - t2) *.5f;
	
	vec4 res;
	res = (*this) * h1;
	res += nextKey *h2;
	res += (nextKey - prevKey) * h3;
	res += (nextKeyP1 - (*this)) * h4;
	res.w = 0.f;
	return  res;
}


float matrix::inverse(const matrix &srcMatrix, bool affine )
{
    float det = 0;

    if(affine)
    {
        det = GetDeterminant();
        float s = 1 / det;
        m[0][0] = (srcMatrix.m[1][1]*srcMatrix.m[2][2] - srcMatrix.m[1][2]*srcMatrix.m[2][1]) * s;
        m[0][1] = (srcMatrix.m[2][1]*srcMatrix.m[0][2] - srcMatrix.m[2][2]*srcMatrix.m[0][1]) * s;
        m[0][2] = (srcMatrix.m[0][1]*srcMatrix.m[1][2] - srcMatrix.m[0][2]*srcMatrix.m[1][1]) * s;
        m[1][0] = (srcMatrix.m[1][2]*srcMatrix.m[2][0] - srcMatrix.m[1][0]*srcMatrix.m[2][2]) * s;
        m[1][1] = (srcMatrix.m[2][2]*srcMatrix.m[0][0] - srcMatrix.m[2][0]*srcMatrix.m[0][2]) * s;
        m[1][2] = (srcMatrix.m[0][2]*srcMatrix.m[1][0] - srcMatrix.m[0][0]*srcMatrix.m[1][2]) * s;
        m[2][0] = (srcMatrix.m[1][0]*srcMatrix.m[2][1] - srcMatrix.m[1][1]*srcMatrix.m[2][0]) * s;
        m[2][1] = (srcMatrix.m[2][0]*srcMatrix.m[0][1] - srcMatrix.m[2][1]*srcMatrix.m[0][0]) * s;
        m[2][2] = (srcMatrix.m[0][0]*srcMatrix.m[1][1] - srcMatrix.m[0][1]*srcMatrix.m[1][0]) * s;
        m[3][0] = -(m[0][0]*srcMatrix.m[3][0] + m[1][0]*srcMatrix.m[3][1] + m[2][0]*srcMatrix.m[3][2]);
        m[3][1] = -(m[0][1]*srcMatrix.m[3][0] + m[1][1]*srcMatrix.m[3][1] + m[2][1]*srcMatrix.m[3][2]);
        m[3][2] = -(m[0][2]*srcMatrix.m[3][0] + m[1][2]*srcMatrix.m[3][1] + m[2][2]*srcMatrix.m[3][2]);
    }
    else
    {
        // transpose matrix
        float src[16];
        for ( int i=0; i<4; ++i )
        {
            src[i]      = srcMatrix.m16[i*4];
            src[i + 4]  = srcMatrix.m16[i*4 + 1];
            src[i + 8]  = srcMatrix.m16[i*4 + 2];
            src[i + 12] = srcMatrix.m16[i*4 + 3];
        }

        // calculate pairs for first 8 elements (cofactors)
        float tmp[12]; // temp array for pairs
        tmp[0]  = src[10] * src[15];
        tmp[1]  = src[11] * src[14];
        tmp[2]  = src[9]  * src[15];
        tmp[3]  = src[11] * src[13];
        tmp[4]  = src[9]  * src[14];
        tmp[5]  = src[10] * src[13];
        tmp[6]  = src[8]  * src[15];
        tmp[7]  = src[11] * src[12];
        tmp[8]  = src[8]  * src[14];
        tmp[9]  = src[10] * src[12];
        tmp[10] = src[8]  * src[13];
        tmp[11] = src[9]  * src[12];

        // calculate first 8 elements (cofactors)
        m16[0] = (tmp[0] * src[5] + tmp[3] * src[6] + tmp[4]  * src[7]) - (tmp[1] * src[5] + tmp[2] * src[6] + tmp[5]  * src[7]);
        m16[1] = (tmp[1] * src[4] + tmp[6] * src[6] + tmp[9]  * src[7]) - (tmp[0] * src[4] + tmp[7] * src[6] + tmp[8]  * src[7]);
        m16[2] = (tmp[2] * src[4] + tmp[7] * src[5] + tmp[10] * src[7]) - (tmp[3] * src[4] + tmp[6] * src[5] + tmp[11] * src[7]);
        m16[3] = (tmp[5] * src[4] + tmp[8] * src[5] + tmp[11] * src[6]) - (tmp[4] * src[4] + tmp[9] * src[5] + tmp[10] * src[6]);
        m16[4] = (tmp[1] * src[1] + tmp[2] * src[2] + tmp[5]  * src[3]) - (tmp[0] * src[1] + tmp[3] * src[2] + tmp[4]  * src[3]);
        m16[5] = (tmp[0] * src[0] + tmp[7] * src[2] + tmp[8]  * src[3]) - (tmp[1] * src[0] + tmp[6] * src[2] + tmp[9]  * src[3]);
        m16[6] = (tmp[3] * src[0] + tmp[6] * src[1] + tmp[11] * src[3]) - (tmp[2] * src[0] + tmp[7] * src[1] + tmp[10] * src[3]);
        m16[7] = (tmp[4] * src[0] + tmp[9] * src[1] + tmp[10] * src[2]) - (tmp[5] * src[0] + tmp[8] * src[1] + tmp[11] * src[2]);

        // calculate pairs for second 8 elements (cofactors)
        tmp[0]  = src[2] * src[7];
        tmp[1]  = src[3] * src[6];
        tmp[2]  = src[1] * src[7];
        tmp[3]  = src[3] * src[5];
        tmp[4]  = src[1] * src[6];
        tmp[5]  = src[2] * src[5];
        tmp[6]  = src[0] * src[7];
        tmp[7]  = src[3] * src[4];
        tmp[8]  = src[0] * src[6];
        tmp[9]  = src[2] * src[4];
        tmp[10] = src[0] * src[5];
        tmp[11] = src[1] * src[4];

        // calculate second 8 elements (cofactors)
        m16[8]  = (tmp[0]  * src[13] + tmp[3]  * src[14] + tmp[4]  * src[15]) - (tmp[1]  * src[13] + tmp[2]  * src[14] + tmp[5]  * src[15]);
        m16[9]  = (tmp[1]  * src[12] + tmp[6]  * src[14] + tmp[9]  * src[15]) - (tmp[0]  * src[12] + tmp[7]  * src[14] + tmp[8]  * src[15]);
        m16[10] = (tmp[2]  * src[12] + tmp[7]  * src[13] + tmp[10] * src[15]) - (tmp[3]  * src[12] + tmp[6]  * src[13] + tmp[11] * src[15]);
        m16[11] = (tmp[5]  * src[12] + tmp[8]  * src[13] + tmp[11] * src[14]) - (tmp[4]  * src[12] + tmp[9]  * src[13] + tmp[10] * src[14]);
        m16[12] = (tmp[2]  * src[10] + tmp[5]  * src[11] + tmp[1]  * src[9])  - (tmp[4]  * src[11] + tmp[0]  * src[9]  + tmp[3]  * src[10]);
        m16[13] = (tmp[8]  * src[11] + tmp[0]  * src[8]  + tmp[7]  * src[10]) - (tmp[6]  * src[10] + tmp[9]  * src[11] + tmp[1]  * src[8]);
        m16[14] = (tmp[6]  * src[9]  + tmp[11] * src[11] + tmp[3]  * src[8])  - (tmp[10] * src[11] + tmp[2]  * src[8]  + tmp[7]  * src[9]);
        m16[15] = (tmp[10] * src[10] + tmp[4]  * src[8]  + tmp[9]  * src[9])  - (tmp[8]  * src[9]  + tmp[11] * src[10] + tmp[5]  * src[8]);

        // calculate determinant
        det = src[0]*m16[0]+src[1]*m16[1]+src[2]*m16[2]+src[3]*m16[3];

        // calculate matrix inverse
        float invdet = 1 / det;
        for ( int j=0; j<16; ++j )
        {
            m16[j] *= invdet;
        }
    }

    return det;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

float matrix::inverse(bool affine)
{
    float det = 0;

    if(affine)
    {
        det = GetDeterminant();
        float s = 1 / det;
        float v00 = (m[1][1]*m[2][2] - m[1][2]*m[2][1]) * s;
        float v01 = (m[2][1]*m[0][2] - m[2][2]*m[0][1]) * s;
        float v02 = (m[0][1]*m[1][2] - m[0][2]*m[1][1]) * s;
        float v10 = (m[1][2]*m[2][0] - m[1][0]*m[2][2]) * s;
        float v11 = (m[2][2]*m[0][0] - m[2][0]*m[0][2]) * s;
        float v12 = (m[0][2]*m[1][0] - m[0][0]*m[1][2]) * s;
        float v20 = (m[1][0]*m[2][1] - m[1][1]*m[2][0]) * s;
        float v21 = (m[2][0]*m[0][1] - m[2][1]*m[0][0]) * s;
        float v22 = (m[0][0]*m[1][1] - m[0][1]*m[1][0]) * s;
        float v30 = -(v00*m[3][0] + v10*m[3][1] + v20*m[3][2]);
        float v31 = -(v01*m[3][0] + v11*m[3][1] + v21*m[3][2]);
        float v32 = -(v02*m[3][0] + v12*m[3][1] + v22*m[3][2]);
        m[0][0] = v00;
        m[0][1] = v01;
        m[0][2] = v02;
        m[1][0] = v10;
        m[1][1] = v11;
        m[1][2] = v12;
        m[2][0] = v20;
        m[2][1] = v21;
        m[2][2] = v22;
        m[3][0] = v30;
        m[3][1] = v31;
        m[3][2] = v32;
    }
    else
    {
        // transpose matrix
        float src[16];
        for ( int i=0; i<4; ++i )
        {
            src[i]      = m16[i*4];
            src[i + 4]  = m16[i*4 + 1];
            src[i + 8]  = m16[i*4 + 2];
            src[i + 12] = m16[i*4 + 3];
        }

        // calculate pairs for first 8 elements (cofactors)
        float tmp[12]; // temp array for pairs
        tmp[0]  = src[10] * src[15];
        tmp[1]  = src[11] * src[14];
        tmp[2]  = src[9]  * src[15];
        tmp[3]  = src[11] * src[13];
        tmp[4]  = src[9]  * src[14];
        tmp[5]  = src[10] * src[13];
        tmp[6]  = src[8]  * src[15];
        tmp[7]  = src[11] * src[12];
        tmp[8]  = src[8]  * src[14];
        tmp[9]  = src[10] * src[12];
        tmp[10] = src[8]  * src[13];
        tmp[11] = src[9]  * src[12];

        // calculate first 8 elements (cofactors)
        m16[0] = (tmp[0] * src[5] + tmp[3] * src[6] + tmp[4]  * src[7]) - (tmp[1] * src[5] + tmp[2] * src[6] + tmp[5]  * src[7]);
        m16[1] = (tmp[1] * src[4] + tmp[6] * src[6] + tmp[9]  * src[7]) - (tmp[0] * src[4] + tmp[7] * src[6] + tmp[8]  * src[7]);
        m16[2] = (tmp[2] * src[4] + tmp[7] * src[5] + tmp[10] * src[7]) - (tmp[3] * src[4] + tmp[6] * src[5] + tmp[11] * src[7]);
        m16[3] = (tmp[5] * src[4] + tmp[8] * src[5] + tmp[11] * src[6]) - (tmp[4] * src[4] + tmp[9] * src[5] + tmp[10] * src[6]);
        m16[4] = (tmp[1] * src[1] + tmp[2] * src[2] + tmp[5]  * src[3]) - (tmp[0] * src[1] + tmp[3] * src[2] + tmp[4]  * src[3]);
        m16[5] = (tmp[0] * src[0] + tmp[7] * src[2] + tmp[8]  * src[3]) - (tmp[1] * src[0] + tmp[6] * src[2] + tmp[9]  * src[3]);
        m16[6] = (tmp[3] * src[0] + tmp[6] * src[1] + tmp[11] * src[3]) - (tmp[2] * src[0] + tmp[7] * src[1] + tmp[10] * src[3]);
        m16[7] = (tmp[4] * src[0] + tmp[9] * src[1] + tmp[10] * src[2]) - (tmp[5] * src[0] + tmp[8] * src[1] + tmp[11] * src[2]);

        // calculate pairs for second 8 elements (cofactors)
        tmp[0]  = src[2] * src[7];
        tmp[1]  = src[3] * src[6];
        tmp[2]  = src[1] * src[7];
        tmp[3]  = src[3] * src[5];
        tmp[4]  = src[1] * src[6];
        tmp[5]  = src[2] * src[5];
        tmp[6]  = src[0] * src[7];
        tmp[7]  = src[3] * src[4];
        tmp[8]  = src[0] * src[6];
        tmp[9]  = src[2] * src[4];
        tmp[10] = src[0] * src[5];
        tmp[11] = src[1] * src[4];

        // calculate second 8 elements (cofactors)
        m16[8]  = (tmp[0]  * src[13] + tmp[3]  * src[14] + tmp[4]  * src[15]) - (tmp[1]  * src[13] + tmp[2]  * src[14] + tmp[5]  * src[15]);
        m16[9]  = (tmp[1]  * src[12] + tmp[6]  * src[14] + tmp[9]  * src[15]) - (tmp[0]  * src[12] + tmp[7]  * src[14] + tmp[8]  * src[15]);
        m16[10] = (tmp[2]  * src[12] + tmp[7]  * src[13] + tmp[10] * src[15]) - (tmp[3]  * src[12] + tmp[6]  * src[13] + tmp[11] * src[15]);
        m16[11] = (tmp[5]  * src[12] + tmp[8]  * src[13] + tmp[11] * src[14]) - (tmp[4]  * src[12] + tmp[9]  * src[13] + tmp[10] * src[14]);
        m16[12] = (tmp[2]  * src[10] + tmp[5]  * src[11] + tmp[1]  * src[9])  - (tmp[4]  * src[11] + tmp[0]  * src[9]  + tmp[3]  * src[10]);
        m16[13] = (tmp[8]  * src[11] + tmp[0]  * src[8]  + tmp[7]  * src[10]) - (tmp[6]  * src[10] + tmp[9]  * src[11] + tmp[1]  * src[8]);
        m16[14] = (tmp[6]  * src[9]  + tmp[11] * src[11] + tmp[3]  * src[8])  - (tmp[10] * src[11] + tmp[2]  * src[8]  + tmp[7]  * src[9]);
        m16[15] = (tmp[10] * src[10] + tmp[4]  * src[8]  + tmp[9]  * src[9])  - (tmp[8]  * src[9]  + tmp[11] * src[10] + tmp[5]  * src[8]);

        // calculate determinant
        det = src[0]*m16[0]+src[1]*m16[1]+src[2]*m16[2]+src[3]*m16[3];

        // calculate matrix inverse
        float invdet = 1 / det;
        for ( int j=0; j<16; ++j )
        {
            m16[j] *= invdet;
        }
    }

    return det;

}


void matrix::rotationAxis(const vec4 & axis, float angle )
{
	float length2 = axis.lengthSq();
	if ( length2 < FLT_EPSILON)
	{
		identity();
		return;
	}

	vec4 n = axis * (1.f / sqrtf(length2));
	float s = sinf(angle);
	float c = cosf(angle);
	float k = 1.f - c;

	float xx = n.x * n.x * k + c;
	float yy = n.y * n.y * k + c;
	float zz = n.z * n.z * k + c;
	float xy = n.x * n.y * k;
	float yz = n.y * n.z * k;
	float zx = n.z * n.x * k;
	float xs = n.x * s;
	float ys = n.y * s;
	float zs = n.z * s;

	m[0][0] = xx;
	m[0][1] = xy + zs;
	m[0][2] = zx - ys;
	m[0][3] = 0.f;
	m[1][0] = xy - zs;
	m[1][1] = yy;
	m[1][2] = yz + xs;
	m[1][3] = 0.f;
	m[2][0] = zx + ys;
	m[2][1] = yz - xs;
	m[2][2] = zz;
	m[2][3] = 0.f;
	m[3][0] = 0.f;
	m[3][1] = 0.f;
	m[3][2] = 0.f;
	m[3][3] = 1.f;
}





void matrix::rotationYawPitchRoll(const float yaw, const float pitch, const float roll )
{
	float cy = cosf(yaw);
	float sy = sinf(yaw);

	float cp = cosf(pitch);
	float sp = sinf(pitch);

	float cr = cosf(roll);
	float sr = sinf(roll);

	float spsy = sp * sy;
	float spcy = sp * cy;

	m[0][0] = cr * cp;
	m[0][1] = sr * cp;
	m[0][2] = -sp;
	m[0][3] = 0;
	m[1][0] = cr * spsy - sr * cy;
	m[1][1] = sr * spsy + cr * cy;
	m[1][2] = cp * sy;
	m[1][3] = 0;
	m[2][0] = cr * spcy + sr * sy;
	m[2][1] = sr * spcy - cr * sy;
	m[2][2] = cp * cy;
	m[2][3] = 0;
	m[3][0] = 0;
	m[3][1] = 0;
	m[3][2] = 0;
	m[3][3] = 1;
}


void matrix::rotationQuaternion( const vec4 &q )
{
	float xx = q.x*q.x;
	float xy = q.x*q.y;
	float xz = q.x*q.z;
	float xw = q.x*q.w;

	float yy = q.y*q.y;
	float yz = q.y*q.z;
	float yw = q.y*q.w;

	float zz = q.z*q.z;
	float zw = q.z*q.w;

	m[0][0] = 1.0f-2.0f*(yy+zz);
	m[0][1] = 2.0f*(xy+zw);
	m[0][2] = 2.0f*(xz-yw);
	m[0][3] = 0.0f;

	m[1][0] = 2.0f*(xy-zw);
	m[1][1] = 1.0f-2.0f*(xx+zz);
	m[1][2] = 2.0f*(yz+xw);
	m[1][3] = 0.0f;

	m[2][0] = 2.0f*(xz+yw);
	m[2][1] = 2.0f*(yz-xw);
	m[2][2] = 1.0f-2.0f*(xx+yy);
	m[2][3] = 0.0f;

	m[3][0] = 0.0f;
	m[3][1] = 0.0f;
	m[3][2] = 0.0f;
	m[3][3] = 1.0f;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

void ZFrustum::NormalizePlane(float frustum[6][4], int side)
{
	// Here we calculate the magnitude of the normal to the plane (point A B C)
	// Remember that (A, B, C) is that same thing as the normal's (X, Y, Z).
	// To calculate magnitude you use the equation:  magnitude = sqrt( x^2 + y^2 + z^2)
	float magnitude = 1.0f / sqrtf( frustum[side][A] * frustum[side][A] +
		frustum[side][B] * frustum[side][B] +
		frustum[side][C] * frustum[side][C] );

	// Then we divide the plane's values by it's magnitude.
	// This makes it easier to work with.
	frustum[side][A] *= magnitude;
	frustum[side][B] *= magnitude;
	frustum[side][C] *= magnitude;
	frustum[side][D] *= magnitude;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
void ZFrustum::Update(const float* clip)
{
	// Now we actually want to get the sides of the frustum.  To do this we take
	// the clipping planes we received above and extract the sides from them.

	// This will extract the RIGHT side of the frustum
	m_Frustum[RIGHT][A] = clip[3] - clip[0];
	m_Frustum[RIGHT][B] = clip[7] - clip[4];
	m_Frustum[RIGHT][C] = clip[11] - clip[8];
	m_Frustum[RIGHT][D] = clip[15] - clip[12];

	// Now that we have a normal (A,B,C) and a Distance (D) to the plane,
	// we want to normalize that normal and Distance.

	// Normalize the RIGHT side
	NormalizePlane(m_Frustum, RIGHT);

	// This will extract the LEFT side of the frustum
	m_Frustum[LEFT][A] = clip[3] + clip[0];
	m_Frustum[LEFT][B] = clip[7] + clip[4];
	m_Frustum[LEFT][C] = clip[11] + clip[8];
	m_Frustum[LEFT][D] = clip[15] + clip[12];

	// Normalize the LEFT side
	NormalizePlane(m_Frustum, LEFT);

	// This will extract the BOTTOM side of the frustum
	m_Frustum[BOTTOM][A] = clip[3] + clip[1];
	m_Frustum[BOTTOM][B] = clip[7] + clip[5];
	m_Frustum[BOTTOM][C] = clip[11] + clip[9];
	m_Frustum[BOTTOM][D] = clip[15] + clip[13];

	// Normalize the BOTTOM side
	NormalizePlane(m_Frustum, BOTTOM);

	// This will extract the TOP side of the frustum
	m_Frustum[TOP][A] = clip[3] - clip[1];
	m_Frustum[TOP][B] = clip[7] - clip[5];
	m_Frustum[TOP][C] = clip[11] - clip[9];
	m_Frustum[TOP][D] = clip[15] - clip[13];

	// Normalize the TOP side
	NormalizePlane(m_Frustum, TOP);

	// This will extract the BACK side of the frustum
	m_Frustum[BACK][A] = clip[3] - clip[2];
	m_Frustum[BACK][B] = clip[7] - clip[6];
	m_Frustum[BACK][C] = clip[11] - clip[10];
	m_Frustum[BACK][D] = clip[15] - clip[14];

	// Normalize the BACK side
	NormalizePlane(m_Frustum, BACK);

	// This will extract the FRONT side of the frustum
	m_Frustum[FRONT][A] = clip[3] + clip[2];
	m_Frustum[FRONT][B] = clip[7] + clip[6];
	m_Frustum[FRONT][C] = clip[11] + clip[10];
	m_Frustum[FRONT][D] = clip[15] + clip[14];

	// Normalize the FRONT side
	NormalizePlane(m_Frustum, FRONT);
}

void ZFrustum::Update(const matrix &view, const matrix& projection)
{
	float   *proj;
	float   *modl;
	float   clip[16];
	//   float   t;

	modl = (float*)view.m16;
	proj = (float*)projection.m16;

	// Now that we have our modelview and projection matrix, if we combine these 2 matrices,
	// it will give us our clipping planes.  To combine 2 matrices, we multiply them.

	clip[ 0] = modl[ 0] * proj[ 0] + modl[ 1] * proj[ 4] + modl[ 2] * proj[ 8] + modl[ 3] * proj[12];
	clip[ 1] = modl[ 0] * proj[ 1] + modl[ 1] * proj[ 5] + modl[ 2] * proj[ 9] + modl[ 3] * proj[13];
	clip[ 2] = modl[ 0] * proj[ 2] + modl[ 1] * proj[ 6] + modl[ 2] * proj[10] + modl[ 3] * proj[14];
	clip[ 3] = modl[ 0] * proj[ 3] + modl[ 1] * proj[ 7] + modl[ 2] * proj[11] + modl[ 3] * proj[15];

	clip[ 4] = modl[ 4] * proj[ 0] + modl[ 5] * proj[ 4] + modl[ 6] * proj[ 8] + modl[ 7] * proj[12];
	clip[ 5] = modl[ 4] * proj[ 1] + modl[ 5] * proj[ 5] + modl[ 6] * proj[ 9] + modl[ 7] * proj[13];
	clip[ 6] = modl[ 4] * proj[ 2] + modl[ 5] * proj[ 6] + modl[ 6] * proj[10] + modl[ 7] * proj[14];
	clip[ 7] = modl[ 4] * proj[ 3] + modl[ 5] * proj[ 7] + modl[ 6] * proj[11] + modl[ 7] * proj[15];

	clip[ 8] = modl[ 8] * proj[ 0] + modl[ 9] * proj[ 4] + modl[10] * proj[ 8] + modl[11] * proj[12];
	clip[ 9] = modl[ 8] * proj[ 1] + modl[ 9] * proj[ 5] + modl[10] * proj[ 9] + modl[11] * proj[13];
	clip[10] = modl[ 8] * proj[ 2] + modl[ 9] * proj[ 6] + modl[10] * proj[10] + modl[11] * proj[14];
	clip[11] = modl[ 8] * proj[ 3] + modl[ 9] * proj[ 7] + modl[10] * proj[11] + modl[11] * proj[15];

	clip[12] = modl[12] * proj[ 0] + modl[13] * proj[ 4] + modl[14] * proj[ 8] + modl[15] * proj[12];
	clip[13] = modl[12] * proj[ 1] + modl[13] * proj[ 5] + modl[14] * proj[ 9] + modl[15] * proj[13];
	clip[14] = modl[12] * proj[ 2] + modl[13] * proj[ 6] + modl[14] * proj[10] + modl[15] * proj[14];
	clip[15] = modl[12] * proj[ 3] + modl[13] * proj[ 7] + modl[14] * proj[11] + modl[15] * proj[15];

	Update(clip);
}

///////////////////////////////////////////////////////////////////////////////////////////////////

bool ZFrustum::OBBInFrustum( const matrix &mt, const vec4 &pos, const vec4& size) const
{
	ZFrustum tmpfrus;
	for (int i=0;i<6;i++)
	{
		((vec4&)tmpfrus.m_Frustum[i]).TransformVector(((vec4&)m_Frustum[i]), mt);
		tmpfrus.m_Frustum[i][3] = m_Frustum[i][3];
	}

	vec4 npos;
	npos.TransformPoint(pos, mt);
	bool bRet = tmpfrus.BoxInFrustum(npos, size);
	return bRet;
}
