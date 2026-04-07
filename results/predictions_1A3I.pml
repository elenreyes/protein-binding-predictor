load /home/nuria/Documents/SBI/Project_1/1A3I.pdb
hide everything
show cartoon
color grey80, all
set cartoon_transparency, 0.3

color None, chain A and resi 3
color None, chain C and resi 65
color None, chain A and resi 4
color None, chain B and resi 31
color None, chain A and resi 7
color None, chain B and resi 34
color None, chain C and resi 61
color None, chain B and resi 35
color None, chain B and resi 32
color None, chain A and resi 5

select predicted_binding, (chain A and resi 3) or (chain C and resi 65) or (chain A and resi 4) or (chain B and resi 31) or (chain A and resi 7) or (chain B and resi 34) or (chain C and resi 61) or (chain B and resi 35) or (chain B and resi 32) or (chain A and resi 5)
show sticks, predicted_binding
show surface, predicted_binding
show surface

show sticks, organic
color green, organic

