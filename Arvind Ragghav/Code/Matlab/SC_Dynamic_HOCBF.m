clc
clear all
close all

% starting point
x0 = [-4;0;0;0;4];
X0 = x0;
uref = [1;0];
T = 7;
dt = 0.001;
N = T/dt;
X = x0;
U = [];
Xobs = [0;0];   % location of the obstacle
r = 2;          % radius of the obstacle
par = 50;

Xc = 0;
Yc = 0;
Xcdot = -1;
Ycdot = 0;
Xcddot = 0;
Ycddot = 0;

xc = Xc;
yc = Yc;
xcdot = Xcdot;
ycdot = Ycdot;
xcddot = Xcddot;
ycddot = Ycddot;

r = 1;
global lr;
lr = 1;
prop = 20;
gamma = 0.1;


for i=1:N
    i
    x = x0(1);
    y = x0(2);
    theta = x0(3);
    w = x0(4);
    v = x0(5);
    
    xc = xc + xcdot*dt;
    yc = yc + ycdot*dt;
    
%     uref = referencecontrol(x,y,theta,w,v,xd,yd,xd_dot,yd_dot,lr);
    h = (x - xc)^2 + (y - yc)^2 - r^2;
    disp(h)
    psi0 = h;
    psi1 = 2*(x - xc)*(v*cos(theta) - v*w*sin(theta)) + 2*(y - yc)*(v*sin(theta) + v*w*cos(theta)) + alpha(prop,psi0) - 2*(x - xc)*xcdot - 2*(y - yc)*ycdot;
    A = [2*(y - yc)*(w*cos(theta) + sin(theta)) + 2*(x - xc)*(cos(theta) - w*sin(theta)),  2*v*(y - yc)*cos(theta) - 2*v*(x - xc)*sin(theta)];
    b1 = 2*(x - xc)*(-v*sin(theta) + v*w*cos(theta))*w + 2*(y - yc)*(v*cos(theta) - v*w*sin(theta))*w;
    b2 = 2*(v*cos(theta) - v*w*sin(theta))*(v*cos(theta) - v*w*sin(theta)) + 2*(v*sin(theta) + v*w*cos(theta))*(v*sin(theta) + v*w*cos(theta));
    b3 = 2*( - xcdot)*(v*cos(theta) - v*w*sin(theta)) + 2*( - ycdot)*(v*sin(theta) + v*w*cos(theta));
    b4 = alphadot(prop,h)*(2*(x - xc)*(v*cos(theta) - v*w*sin(theta)) + 2*(y - yc)*(v*sin(theta) + v*w*cos(theta)) - 2*(x - xc)*xcdot - 2*(y - yc)*ycdot);
    b5 = - 2*(v*cos(theta) - v*w*sin(theta))*xcdot - 2*(v*sin(theta) + v*w*cos(theta))*ycdot + 2*xcdot*xcdot + 2*ycdot*ycdot - 2*(x - xc)*xcddot - 2*(y - yc)*ycddot ;
    b6 = alpha(prop,psi1) + A*uref;
    b = b1 + b2 + b3 + b4 + b5 + b6;
%     b = (alphadot(prop,h)*(- 2*(x - xc)*xcdot - 2*(y - yc)*ycdot))+(2*( -xcdot)*(v*cos(theta) - v*w*sin(theta)) + 2*( -ycdot)*(v*sin(theta) - v*w*cos(theta))  - 2*( - xcdot)*xcdot - 2*( - ycdot)*ycdot + - 2*(x - xc)*xcddot - 2*(y - yc)*ycddot)+(v*w*cos(theta) + v*sin(theta))*(2*(y-yc)*(alphadot(prop,h)) + 2*(v*w*cos(theta) + v*sin(theta))) + (v*cos(theta) - v*w*sin(theta))*(2*(x - xc)*alphadot(prop,h) + 2*(v*cos(theta) - v*w*sin(theta))) + v*w*(2*(x - xc)*(-v*w*cos(theta) - v*sin(theta)) + 2*(y - yc)*(v*cos(theta) - v*w*sin(theta))) + alpha(prop,psi1) + A*uref;
%     disp(A)
%     disp(b)
    
    % finding the  minimally evasive control law
    H = eye(2);
    f = [0;0];
    delta = quadprog(H,f,-A,b);
    
    f = [v*cos(theta) - v*w*sin(theta);
        v*sin(theta) + v*w*cos(theta);
        v*w/lr;
        0;
        0];
    g = [0,0;
        0,0;
        0,0;
        0,1;
        1,0];
    
    u = uref + delta;
    x0 = x0 + (f + g*u)*dt;
    X = [X,x0];
    U = [U,u];
end
%%
close all
figure(1)
theta = linspace(0,2*pi,100);

for i=1:N
    i
    plot(X(1,1:i),X(2,1:i),'b','LineWidth',2);
    hold on 
    plot(Xc + Xcdot*i*dt + r*cos(theta),Yc + Ycdot*i*dt + r*sin(theta),'k','LineWidth',1.8);
    hold on
    scatter(Xc + Xcdot*i*dt,Yc + Ycdot*i*dt,'k','filled')
    hold on 
    scatter(X(1,i),X(2,i),'r','filled')
    hold off
    axis([-5,5,-5,5])
    legend('system trajectory','obstacle','current position of the vehicle')
    grid on
    pause(0.000001)
end

%%
function k = alpha(prop,z)
   k = prop*z;
end

%%
function k = alphadot(prop,z)
    k = prop;
end

% %% reference control
% function uref = referencecontrol(x,y,theta,w,v,xd,yd,xd_dot,yd_dot,lr)
%     xdot = (v*cos(theta) - w*v*sin(theta));
%     ydot = (v*sin(theta) + w*v*cos(theta));
%     B = [cos(theta) - v*w*sin(theta), -v*cos(theta);
%         sin(theta) + w*v*cos(theta), -v*sin(theta)];
%     ff = [(-v*sin(theta) - v*w*cos(theta))*v*w/lr;
%           (v*cos(theta) - v*w*sin(theta))*v*w/lr];
%     K = [30,11];
%     mu = [K*[(x - xd);(xdot - xd_dot)]; K*[(y-yd);(ydot-yd_dot)]];
%     uref = inv(B)*(-ff + mu);
%     
% end
