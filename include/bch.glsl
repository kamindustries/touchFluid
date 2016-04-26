///////////////////////////////////////////////////////////////////////
// B C H
///////////////////////////////////////////////////////////////////////
vec3 rgb2def(vec3 _col){
  mat3 XYZ; // Adobe RGB (1998)
  XYZ[0] = vec3(0.5767309, 0.1855540, 0.1881852);
  XYZ[1] = vec3(0.2973769, 0.6273491, 0.0752741);
  XYZ[2] = vec3(0.0270343, 0.0706872, 0.9911085);
  mat3 DEF;
  DEF[0] = vec3(0.2053, 0.7125, 0.4670);
  DEF[1] = vec3(1.8537, -1.2797, -0.4429);
  DEF[2] = vec3(-0.3655, 1.0120, -0.6104);
  vec3 xyz = _col.rgb * XYZ;
  vec3 def = xyz * DEF;
  return def;
}

vec3 def2rgb(vec3 _def){
  mat3 XYZ;
  XYZ[0] = vec3(0.6712, 0.4955, 0.1540);
  XYZ[1] = vec3(0.7061, 0.0248, 0.5223);
  XYZ[2] = vec3(0.7689, -0.2556, -0.8645);
  mat3 RGB; // Adobe RGB (1998)
  RGB[0] = vec3(2.0413690, -0.5649464, -0.3446944);
  RGB[1] = vec3(-0.9692660, 1.8760108, 0.0415560);
  RGB[2] = vec3(0.0134474, -0.1183897, 1.0154096);
  vec3 xyz = _def * XYZ;
  vec3 rgb = xyz * RGB;
  return rgb;
}

// Get B, C, H
float getB(vec3 _def){
    return sqrt((_def.r*_def.r) + (_def.g*_def.g) + (_def.b*_def.b));
}
float getC(vec3 _def){
    vec3 def_D = vec3(1.,0.,0.);
    return atan(length(cross(_def,def_D)), dot(_def,def_D));
}
float getH(vec3 _def){
    vec3 def_E_axis = vec3(0.,1.,0.);
    return atan(_def.z, _def.y) - atan(def_E_axis.z, def_E_axis.y) ;
}

vec3 rgb2bch(vec3 _col){
  vec3 DEF = rgb2def(_col);
  return vec3(getB(DEF), getC(DEF), getH(DEF));
}

vec3 bch2rgb(vec3 _bch){
  vec3 def;
  def.x = _bch.x * cos(_bch.y);
  def.y = _bch.x * sin(_bch.y) * cos(_bch.z);
  def.z = _bch.x * sin(_bch.y) * sin(_bch.z);
  return def2rgb(def);
}
//BCH
