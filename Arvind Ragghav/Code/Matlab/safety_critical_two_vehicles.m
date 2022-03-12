%% code for Safety critical control implementation: 
% moving obstacle with the object travelling towards a final point
clc;
clear all;
close all;

r = 1;
dt = 0.01;
t = 0:dt:3;

% initial conditions for object 2
C_0 = [4;4];
C_f = [-2;3];
Uc = [];  

% initial conditions for object 1
X_0 = [0;0];      
X_f = [0;6];
Ux = [] ;    % array for storing the actual control inputs 

alphax = @(k) 10*tanh(k);          % function that is used to tell how to react to the obstacle for the first vehicle
alphac = @(k) 2*tanh(k);           % function that is used to tell how to react to the obstacle for the second vehicle

X = X_0;
C = C_0;

Hhx = [];
Hhc = [];

errorx = 10;
errorc = 10;

epsilon = 1e-1;
X_pos = X_0;
C_pos = C_0;

p = figure;
axis tight manual % this ensures that getframe() returns a consistent size
filename = 'SafetyCriticaltwovehicles.gif';
i = 1;

while errorx > epsilon || errorc > epsilon
    % current positions of the two objects
    x = X_pos(1,1);
    y = X_pos(2,1);
    cx = C_pos(1,1);
    cy = C_pos(2,1);
    
    % defining the target velocities for the two obstacles
    u_ref_x = (X_f - X_pos);
    u_ref_c = (C_f - C_pos);
    
    
    % quadprog matrices for the first object
    Hx = eye(2,2);           
    fx = [0;0];
    hx =  sqrt((x - cx)^2 +  (y - cy)^2) - r^2 ;
    Ax = [(x - cx)/sqrt((x - cx)^2 +  (y - cy)^2), (y-cy)/sqrt((x - cx)^2 +  (y - cy)^2)]; 
    
    Hhx = [Hhx, hx];
    bx = (alphax(hx) + Ax*u_ref_x - Ax*u_ref_c );
    
    deltax = quadprog(Hx,fx,-Ax,bx);
    ux = u_ref_x + deltax;
    Ux = [Ux, ux];
    X_pos = X_pos + (ux)*dt;
    X = [X, X_pos];
    
    % quadprog matrices for the second object
    Hc = eye(2,2);           
    fc = [0;0];
    hc =  sqrt((x - cx)^2 +  (y - cy)^2) - r^2 ;
    Ac = [(cx - x)/sqrt((x - cx)^2 +  (y - cy)^2), (cy - y)/sqrt((x - cx)^2 +  (y - cy)^2)]; 
    
    Hhc = [Hhc, hc];
    bc = (alphax(hc) + Ac*u_ref_c - Ac*u_ref_x );
    
    deltac = quadprog(Hc,fc,-Ac,bc);
    uc = u_ref_c + deltac;
    Uc = [Uc, uc];
    C_pos = C_pos + (uc)*dt;
    C = [C, C_pos];
    
    % calculating the error from the final position
    errorx =  norm(X_f - X_pos)
    errorc =  norm(C_f - C_pos)
    
    theta = 0:0.01:2*pi;
    plot(X(1,:),X(2,:));
    hold on 
    plot(r*cos(theta) + x, r*sin(theta) + y)
    hold on 
    plot(C(1,:),C(2,:))
    hold on
    plot(r*cos(theta) + cx, r*sin(theta)+cy);
    hold off
    title("object 1 vs object 2, time: " + dt*i + "")
    legend('path of object 1','region of influence of object 1','path of object 2', 'region of influence of object 2')
    axis([-6,6,-6,6])
    
    frame = getframe(p ); 
    im = frame2im(frame); 
    [imind,cm] = rgb2ind(im,256); 
    % Write to the GIF File 
    if i == 1 
        imwrite(imind,cm,filename,'gif', 'Loopcount',inf); 
    else 
        imwrite(imind,cm,filename,'gif','WriteMode','append'); 
    end 
    i = i+1;
end


figure 
subplot(2,1,1)
plot(1:dt:(length(Ux)-1)*dt, Ux(1,:))
title('Input u1')
legend('ux')
subplot(2,1,2)
plot(1:dt:(length(Ux)-1)*dt, Ux(2,:))
title('Input u2')
legend('uy')

figure 
subplot(2,1,1)
plot(1:dt:(length(Uc)-1)*dt, Uc(1,:))
title('Input u1')
legend('uc_x')
subplot(2,1,2)
plot(1:dt:(length(Uc)-1)*dt, Uc(2,:))
title('Input u2')
legend('uc_y')

figure 
subplot(2,1,1)
plot(1:dt:(length(Ux))*dt, X(1,:))
title('x vs t for object 1')
subplot(2,1,2)
plot(1:dt:(length(Ux))*dt,X(2,:))
title('y vs t for object 1')



figure 
subplot(2,1,1)
plot(1:dt:(length(Uc))*dt, C(1,:))
title('x vs t for object 2')
subplot(2,1,2)
plot(1:dt:(length(Uc))*dt,C(2,:))
title('y vs t for object 2')