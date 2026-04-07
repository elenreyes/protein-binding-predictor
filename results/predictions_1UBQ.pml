load /home/nuria/Documents/SBI/Project_1/1UBQ.pdb
hide everything
show cartoon
color grey80, all
set cartoon_transparency, 0.3

color yellow, chain A and resi 8
color yellow, chain A and resi 44
color yellow, chain A and resi 73
color yellow, chain A and resi 4
color yellow, chain A and resi 59
color yellow, chain A and resi 27
color yellow, chain A and resi 36
color yellow, chain A and resi 49
color None, chain A and resi 7
color None, chain A and resi 68

select predicted_binding, (chain A and resi 8) or (chain A and resi 44) or (chain A and resi 73) or (chain A and resi 4) or (chain A and resi 59) or (chain A and resi 27) or (chain A and resi 36) or (chain A and resi 49) or (chain A and resi 7) or (chain A and resi 68)
show sticks, predicted_binding
show surface, predicted_binding
show surface

show sticks, organic
color green, organic

