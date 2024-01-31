\* Source Pyomo model name=unknown *\

min 
obj:
+0.07311000061035157 chargepower(0)
+0.07072000122070313 chargepower(1)
+0.06908999633789062 chargepower(2)
+0.05804000091552734 chargepower(3)
+0.05990999984741211 chargepower(4)
+0.056 chargepower(5)
+0.04997000122070312 chargepower(6)
+0.0472599983215332 chargepower(7)
+0.04811999893188477 chargepower(8)
+0.04797000122070313 chargepower(9)
+0.05158000183105469 chargepower(10)
+0.06006000137329102 chargepower(11)
+0.08033999633789063 chargepower(12)
+0.09051000213623046 chargepower(13)
+0.09920999908447266 chargepower(14)

s.t.

c_e_chargepower_rule(0)_:
+1 chargepower(0)
-0.692 chargecurrent(0)
= 0

c_e_chargepower_rule(1)_:
+1 chargepower(1)
-0.692 chargecurrent(1)
= 0

c_e_chargepower_rule(2)_:
+1 chargepower(2)
-0.692 chargecurrent(2)
= 0

c_e_chargepower_rule(3)_:
+1 chargepower(3)
-0.692 chargecurrent(3)
= 0

c_e_chargepower_rule(4)_:
+1 chargepower(4)
-0.692 chargecurrent(4)
= 0

c_e_chargepower_rule(5)_:
+1 chargepower(5)
-0.692 chargecurrent(5)
= 0

c_e_chargepower_rule(6)_:
+1 chargepower(6)
-0.692 chargecurrent(6)
= 0

c_e_chargepower_rule(7)_:
+1 chargepower(7)
-0.692 chargecurrent(7)
= 0

c_e_chargepower_rule(8)_:
+1 chargepower(8)
-0.692 chargecurrent(8)
= 0

c_e_chargepower_rule(9)_:
+1 chargepower(9)
-0.692 chargecurrent(9)
= 0

c_e_chargepower_rule(10)_:
+1 chargepower(10)
-0.692 chargecurrent(10)
= 0

c_e_chargepower_rule(11)_:
+1 chargepower(11)
-0.692 chargecurrent(11)
= 0

c_e_chargepower_rule(12)_:
+1 chargepower(12)
-0.692 chargecurrent(12)
= 0

c_e_chargepower_rule(13)_:
+1 chargepower(13)
-0.692 chargecurrent(13)
= 0

c_e_chargepower_rule(14)_:
+1 chargepower(14)
-0.692 chargecurrent(14)
= 0

c_l_energy_:
+1 chargepower(0)
+1 chargepower(1)
+1 chargepower(2)
+1 chargepower(3)
+1 chargepower(4)
+1 chargepower(5)
+1 chargepower(6)
+1 chargepower(7)
+1 chargepower(8)
+1 chargepower(9)
+1 chargepower(10)
+1 chargepower(11)
+1 chargepower(12)
+1 chargepower(13)
+1 chargepower(14)
>= 40

c_u_max_power(0)_:
+1 chargepower(0)
<= 17.7152

c_u_max_power(1)_:
+1 chargepower(1)
<= 17.7152

c_u_max_power(2)_:
+1 chargepower(2)
<= 17.7152

c_u_max_power(3)_:
+1 chargepower(3)
<= 17.7152

c_u_max_power(4)_:
+1 chargepower(4)
<= 17.7152

c_u_max_power(5)_:
+1 chargepower(5)
<= 17.7152

c_u_max_power(6)_:
+1 chargepower(6)
<= 17.7152

c_u_max_power(7)_:
+1 chargepower(7)
<= 17.7152

c_u_max_power(8)_:
+1 chargepower(8)
<= 17.7152

c_u_max_power(9)_:
+1 chargepower(9)
<= 17.7152

c_u_max_power(10)_:
+1 chargepower(10)
<= 17.7152

c_u_max_power(11)_:
+1 chargepower(11)
<= 17.7152

c_u_max_power(12)_:
+1 chargepower(12)
<= 17.7152

c_u_max_power(13)_:
+1 chargepower(13)
<= 17.7152

c_u_max_power(14)_:
+1 chargepower(14)
<= 17.7152

bounds
   0 <= chargepower(0) <= +inf
   0 <= chargepower(1) <= +inf
   0 <= chargepower(2) <= +inf
   0 <= chargepower(3) <= +inf
   0 <= chargepower(4) <= +inf
   0 <= chargepower(5) <= +inf
   0 <= chargepower(6) <= +inf
   0 <= chargepower(7) <= +inf
   0 <= chargepower(8) <= +inf
   0 <= chargepower(9) <= +inf
   0 <= chargepower(10) <= +inf
   0 <= chargepower(11) <= +inf
   0 <= chargepower(12) <= +inf
   0 <= chargepower(13) <= +inf
   0 <= chargepower(14) <= +inf
   0 <= chargecurrent(0) <= 32
   0 <= chargecurrent(1) <= 32
   0 <= chargecurrent(2) <= 32
   0 <= chargecurrent(3) <= 32
   0 <= chargecurrent(4) <= 32
   0 <= chargecurrent(5) <= 32
   0 <= chargecurrent(6) <= 32
   0 <= chargecurrent(7) <= 32
   0 <= chargecurrent(8) <= 32
   0 <= chargecurrent(9) <= 32
   0 <= chargecurrent(10) <= 32
   0 <= chargecurrent(11) <= 32
   0 <= chargecurrent(12) <= 32
   0 <= chargecurrent(13) <= 32
   0 <= chargecurrent(14) <= 32
end
