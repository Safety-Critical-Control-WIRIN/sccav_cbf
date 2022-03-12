%% code for Safety critical control implementation: 
% static object with the object travelling towards a final point
clc;
clear all;
close all;

r = 1;
dt = 0.01;
t = 0:dt:3;

% center of the 2 obstacles
cx1 = 2;
cy1 = 2;
cx2 = 2;
cy2 = 2.5;

X_0 = [2;0];                % initial condition
X_f = [1.7;6];
U = [] ;    % array for storing the actual control inputs 

alpha1 = @(k) 5*tanh(k);          % function that is used to tell how to react to the obstacle
alpha2 = @(k) 5*tanh(k);          % function that is used to tell how to react to the obstacle

X = X_0;
Hh = zeros(1,length(t)-1);
error = 10;
epsilon = 1e-1;
X_pos = X_0;


p = figure;
axis tight manual % this ensures that getframe() returns a consistent size
filename = 'SafetyCriticalmultipleStaticObstacles.gif';
i = 1;
while error > epsilon
    x = X_pos(1,1);
    y = X_pos(2,1);
    
    u_ref = 1*(X_f - X_pos);
    
    % defining quadprog matrices
    H = eye(2,2);           
    f = [0;0];
    
    h1 =  sqrt((x - cx1)^2 +  (y - cy1)^2) - r^2 ;
    A1 = [(x - cx1)/sqrt((x - cx1)^2 +  (y - cy1)^2), (y-cy1)/sqrt((x - cx1)^2 +  (y - cy1)^2)]; 
    
    h2 =  sqrt((x - cx2)^2 +  (y - cy2)^2) - r^2 ;
    A2 = [(x - cx2)/sqrt((x - cx2)^2 +  (y - cy2)^2), (y-cy2)/sqrt((x - cx2)^2 +  (y - cy2)^2)]; 
    A = [A1;A2];
    Hh = [Hh, h1 , h2];
    b = [(alpha1(h1) + A1*u_ref );(alpha2(h2) + A2*u_ref)];
    
    delta = quadprog(H,f,-A,b);
    u = u_ref + delta;
    U = [U, u];
    X_pos = X_pos + (u)*dt;
    X = [X, X_pos];
    error = norm(X_f - X_pos)
    
    theta = 0:0.01:2*pi;
    plot(X(1,:),X(2,:));
    hold on 
    plot(r*cos(theta) + cx1, r*sin(theta)+cy1);
    hold on 
    plot(r*cos(theta) + cx2, r*sin(theta)+cy2);
    hold off
    title('System vs static obstacle')
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