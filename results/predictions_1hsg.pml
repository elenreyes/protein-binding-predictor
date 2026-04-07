load /home/nuria/Documents/SBI/Project_1/1hsg.pdb
hide everything
show cartoon
color grey80, all
set cartoon_transparency, 0.3

color orange, chain B and resi 27
color orange, chain A and resi 28
color orange, chain A and resi 42
color orange, chain A and resi 27
color orange, chain B and resi 42
color orange, chain B and resi 84
color orange, chain A and resi 84
color orange, chain B and resi 28
color orange, chain B and resi 6
color yellow, chain B and resi 25
color yellow, chain A and resi 40
color yellow, chain A and resi 32
color yellow, chain A and resi 25
color yellow, chain A and resi 83
color yellow, chain A and resi 6
color yellow, chain A and resi 48
color yellow, chain B and resi 32
color yellow, chain B and resi 67
color yellow, chain A and resi 8
color yellow, chain B and resi 78
color yellow, chain A and resi 1
color yellow, chain B and resi 49
color yellow, chain B and resi 48

select predicted_binding, (chain B and resi 27) or (chain A and resi 28) or (chain A and resi 42) or (chain A and resi 27) or (chain B and resi 42) or (chain B and resi 84) or (chain A and resi 84) or (chain B and resi 28) or (chain B and resi 6) or (chain B and resi 25) or (chain A and resi 40) or (chain A and resi 32) or (chain A and resi 25) or (chain A and resi 83) or (chain A and resi 6) or (chain A and resi 48) or (chain B and resi 32) or (chain B and resi 67) or (chain A and resi 8) or (chain B and resi 78) or (chain A and resi 1) or (chain B and resi 49) or (chain B and resi 48)
show sticks, predicted_binding
show surface, predicted_binding
show surface

show sticks, organic
color green, organic

