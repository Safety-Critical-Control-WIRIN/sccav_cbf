%% code for Safety critical control implementation: 
% dynamic object with both the suject and obstacle moving in straight lines
clc;
clear all;
close all;

dt = 0.01;
t = 0:dt:3;
c_0 = [4;4];
C0 = zeros(2,length(t));
C0(:,1) = c_0;
u_obstacle  = [-4;0];              % defining obstacle control

u_ref       = [0;2.5];              % defining reference control
X_0         = [0;0];                % initial condition
U           = zeros(2, length(t)-1);% array for storing the actual control inputs 
X           = zeros(2, length(t));
gamma       = @(k) 50*tanh(k);         % function that is used to tell how to react to the obstacle

X(:,1) = X_0;
Hh = zeros(1,length(t)-1);


p = figure;
axis tight manual % this ensures that getframe() returns a consistent size
filename = 'SafetyCriticalDynamic.gif';

for i=1:length(t)-1
    
    x = X(1,i);               % current x position
    y = X(2,i);               % current y coordinate
    
    % defining quadprog matrices
    H = eye(2,2);           
    f = [0;0];
    A = -[2*(x - C0(1,i)), 2*(y - C0(2,i))]; 
    h =  ((x - C0(1,i))^2 +  (y - C0(2,i))^2 - 1) ;
    Hh(1,i) = h;
    b = (gamma(h) + A*u_ref);
    
    delta = quadprog(H,f,A,b);
    u = u_ref + delta;
    U(:,i) = u;
    X(:,i+1) = X(:,i) + (u + u_ref)*dt;
    C0(:,i+1) = C0(:,i) + (u_obstacle)*dt;
    
    
    theta = 0:0.01:2*pi;
    plot(X(1,1:i),X(2,1:i));
    hold on 
    plot(cos(theta) + C0(1,i), sin(theta) + C0(2,i));
    hold off
    title("time: " +  i*dt + " " )
    axis([-6,6,-6,6]);
    
    
    frame = getframe(p ); 
    im = frame2im(frame); 
    [imind,cm] = rgb2ind(im,256); 
    % Write to the GIF File 
    if i == 1 
        imwrite(imind,cm,filename,'gif', 'Loopcount',inf); 
    else 
        imwrite(imind,cm,filename,'gif','WriteMode','append'); 
    end 
end
figure 
title('input vs time')
plot(t(1:length(t)-1), U(1,:))
hold on 
plot(t(1:length(t)-1), U(2,:))
hold off
legend('ux', 'uy')

figure 
subplot(2,1,1)
plot(t, X(1,:))
subplot(2,1,2)
plot(t,X(2,:))
title('individual coordinates vs time')

figure 
plot(t(1:length(t)-1), Hh)