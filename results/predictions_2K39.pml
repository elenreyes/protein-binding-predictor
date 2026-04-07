load /home/nuria/Documents/SBI/Project_1/2K39.pdb
hide everything
show cartoon
color grey80, all
set cartoon_transparency, 0.3

color yellow, chain A and resi 9
color yellow, chain A and resi 76
color yellow, chain A and resi 47
color yellow, chain A and resi 73
color yellow, chain A and resi 75
color yellow, chain A and resi 8
color yellow, chain A and resi 62
color yellow, chain A and resi 1
color yellow, chain A and resi 4
color yellow, chain A and resi 10

select predicted_binding, (chain A and resi 9) or (chain A and resi 76) or (chain A and resi 47) or (chain A and resi 73) or (chain A and resi 75) or (chain A and resi 8) or (chain A and resi 62) or (chain A and resi 1) or (chain A and resi 4) or (chain A and resi 10)
show sticks, predicted_binding
show surface, predicted_binding
show surface

show sticks, organic
color green, organic

