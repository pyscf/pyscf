Pilm(z,l,m) := sqrt((l-m)!/(l+m)!)*sum((-1)**k * 2**(-l) * binomial(l,k) * binomial(2*l-2*k, l) * ((l-2*k)!)/((l-2*k-m)!) * r**(2*k)* z**(l-2*k-m), k, 0, (l-m)/2)$

rsh(x,y,z,l,m) := 
  if m>0 then 
    sqrt((2*l+1)/(4*%pi))*sum(binomial(m,p) * x**p *y**(m-p) * cos((m-p)*%pi/2), p, 0, m)*Pilm(z,l,m)/r**l*sqrt(2)
  else if m<0 then
    sqrt((2*l+1)/(4*%pi))*sum(binomial(abs(m),p) * x**p * y**(abs(m)-p) * sin((abs(m)-p)*%pi/2), p, 0, abs(m))*Pilm(z,l,abs(m))/r**l*sqrt(2)
  else if m=0 then
    sqrt((2*l+1)/(4*%pi))*Pilm(z,l,m)/r**l$

load(f90);
:lisp (setq *f90-output-line-length-max* 1000000000)
declare(l, integer);
declare(m, integer);

for l: 0 thru 5 do
  for m: -l thru l do 
    f90([l,m], ev( subst(r=1, rsh(x,y,z,l,m) ), trigsimp) );

