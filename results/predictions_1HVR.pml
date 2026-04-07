load /home/nuria/Documents/SBI/Project_1/1HVR.pdb
hide everything
show cartoon
color grey80, all
set cartoon_transparency, 0.3

color red, chain B and resi 25
color red, chain A and resi 25
color orange, chain B and resi 42
color orange, chain A and resi 6
color orange, chain A and resi 42
color orange, chain B and resi 27
color orange, chain B and resi 6
color orange, chain A and resi 83
color yellow, chain A and resi 27
color yellow, chain B and resi 28
color yellow, chain A and resi 28
color yellow, chain B and resi 83
color yellow, chain B and resi 96
color yellow, chain B and resi 99
color yellow, chain A and resi 10
color yellow, chain B and resi 84
color yellow, chain A and resi 88
color yellow, chain A and resi 96
color yellow, chain A and resi 95
color yellow, chain A and resi 99
color yellow, chain B and resi 40
color yellow, chain B and resi 53
color yellow, chain B and resi 23

select predicted_binding, (chain B and resi 25) or (chain A and resi 25) or (chain B and resi 42) or (chain A and resi 6) or (chain A and resi 42) or (chain B and resi 27) or (chain B and resi 6) or (chain A and resi 83) or (chain A and resi 27) or (chain B and resi 28) or (chain A and resi 28) or (chain B and resi 83) or (chain B and resi 96) or (chain B and resi 99) or (chain A and resi 10) or (chain B and resi 84) or (chain A and resi 88) or (chain A and resi 96) or (chain A and resi 95) or (chain A and resi 99) or (chain B and resi 40) or (chain B and resi 53) or (chain B and resi 23)
show sticks, predicted_binding
show surface, predicted_binding
show surface

show sticks, organic
color green, organic

