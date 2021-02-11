function [mu, V, K]=KalmanFilter(x,mu0,L0,A,B, Gamma, Sigma)


%(c) 2017, Georgia Koppe, Dept. Theoretical Neuroscience, CIMH, Heidelberg
%University
%for comments, questions, errors, please contact
%Georgia.Koppe@zi-mannhiem.de


% x: observed variables (p x n, with p= number obs., n= number time steps)
% A: transition matrix
% C: control variable matrix
% u: control variables
% B: observation/emission matrix
% Gamma: observation/emission covariance
% Sigma: transition covariance
% mu0, L0: initial values

p= size(Sigma, 1); %dimensionality of latent states
n= size(x,2);      %number of time steps
q= size(x,1);      %number of observations

% are control variables entered? if not, set to 0
if nargin<8
    C=zeros(p,q); u=zeros(q,n);
end

%initialize variables for filter and smoother
L = zeros(p,p,n);           % measurement covariance matrix
L(:,:,1)=L0;                % prior covariance
mu_p  = zeros(p,n);         % predicted expected value
mu_p(:,1)=mu0;              % prior expected value
mu   = zeros(p,n)  ;        % filter expected value
V   = zeros(p,p,n);         % filter covariance matrix
K      = zeros(p,q,n);      % Kalman Gain

%KALMAN FILTER 
%--------------------------------------------------------------------------
%first step
K(:,:,1)    = L(:,:,1)*B'/(B*L(:,:,1)*B'+Gamma);            %Kalman gain
mu(:,1)     = mu_p(:,1)+K(:,:,1)*(x(:,1)-B*mu_p(:,1));
V(:,:,1)    = (eye(p)-K(:,:,1)*B)*L(:,:,1);

for t = 2:n %go forwards
    L(:,:,t)    = A*V(:,:,t-1)*A'+Sigma;
    K(:,:,t)    = L(:,:,t)*B'/(B*L(:,:,t)*B'+Gamma);        %Kalman gain
    mu_p(:,t)   = A*mu(:,t-1)+C*u(:,t);                     %model prediction
    mu(:,t)     = mu_p(:,t) + K(:,:,t)*(x(:,t)-B*mu_p(:,t));%filtered state 
    V(:,:,t)    = (eye(p) - K(:,:,t)*B)*L(:,:,t);           %filtered covariance 
end
