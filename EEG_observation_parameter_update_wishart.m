function [Qu_post,Qx_post,RePst,vPst, cost] = EEG_observation_parameter_update_wishart(data,mutT,VtT,Re0,v0,init_Qu,init_Qx)

yDim = size(data.EEG,1);
T = size(data.EEG,2);
y = data.EEG;
C = data.L*data.G;
Phi = Re0*(v0-yDim)+(y-C*mutT)*(y-C*mutT)'+C*sum(VtT,3)*C';
vPst = v0+T;
RePst  = Phi/(vPst-yDim);
Qu_post = init_Qu;
Qx_post = init_Qx;
cost = vPst*(yDim*log(vPst)+log(det(Phi/vPst)))+trace(RePst\(y-C*mutT)*(y-C*mutT)'+C*sum(VtT,3)*C')*(vPst/(vPst-yDim))-yDim*vPst;