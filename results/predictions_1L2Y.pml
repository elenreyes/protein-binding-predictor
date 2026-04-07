load /home/nuria/Documents/SBI/Project_1/1L2Y.pdb
hide everything
show cartoon
color grey80, all
set cartoon_transparency, 0.3

color yellow, chain A and resi 2
color yellow, chain A and resi 13
color yellow, chain A and resi 15
color None, chain A and resi 3
color None, chain A and resi 17
color None, chain A and resi 5
color None, chain A and resi 19
color None, chain A and resi 6
color None, chain A and resi 16
color None, chain A and resi 20

select predicted_binding, (chain A and resi 2) or (chain A and resi 13) or (chain A and resi 15) or (chain A and resi 3) or (chain A and resi 17) or (chain A and resi 5) or (chain A and resi 19) or (chain A and resi 6) or (chain A and resi 16) or (chain A and resi 20)
show sticks, predicted_binding
show surface, predicted_binding
show surface

show sticks, organic
color green, organic

