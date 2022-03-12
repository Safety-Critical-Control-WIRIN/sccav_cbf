%% code for Safety critical control implementation: 
% moving obstacle with the object travelling towards a final point
clc;
clear all;
close all;

r = 1;
dt = 0.01;
t = 0:dt:3;
c_0 = [4;4];
C_pos = c_0;
u_obstacle  = [-4;0];  

X_0 = [0;0];                % initial condition
X_f = [0;6];
U = [] ;    % array for storing the actual control inputs 

alpha = @(k) 10*tanh(k);          % function that is used to tell how to react to the obstacle

X = X_0;
Hh = [];
error = 10;
epsilon = 1e-1;
X_pos = X_0;


p = figure;
axis tight manual % this ensures that getframe() returns a consistent size
filename = 'SafetyCriticalStaticEndpoint.gif';
i = 1;
while error > epsilon
    x = X_pos(1,1);
    y = X_pos(2,1);
    cx = C_pos(1,1);
    cy = C_pos(2,1);
    
   
    u_ref = (X_f - X_pos);
    
    % defining quadprog matrices
    H = eye(2,2);           
    f = [0;0];
    
    h =  sqrt((x - cx)^2 +  (y - cy)^2) - r^2 ;
    A = [(x - cx)/sqrt((x - cx)^2 +  (y - cy)^2), (y-cy)/sqrt((x - cx)^2 +  (y - cy)^2)]; 
    
    Hh = [Hh, h];
    b = (alpha(h) + A*u_ref - A*u_obstacle );
    
    delta = quadprog(H,f,-A,b);
    u = u_ref + delta;
    U = [U, u];
    X_pos = X_pos + (u)*dt;
    C_pos = C_pos + (u_obstacle)*dt
    
    
    X = [X, X_pos];
    error = norm(X_f - X_pos)
    
    theta = 0:0.01:2*pi;
    plot(X(1,:),X(2,:));
    hold on 
    plot(r*cos(theta) + cx, r*sin(theta)+cy);
    hold off
    title("System vs static obstacle, time: " + dt*i + "")
    legend('path of the system', 'Static Obstacle')
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
plot(1:dt:(length(U)-1)*dt, U(1,:))
title('Input u1')
legend('ux')
subplot(2,1,2)
plot(1:dt:(length(U)-1)*dt, U(2,:))
title('Input u2')
legend('uy')


figure 
subplot(2,1,1)
plot(1:dt:(length(U))*dt, X(1,:))
title('x vs t')
subplot(2,1,2)
plot(1:dt:(length(U))*dt,X(2,:))
title('y vs t')