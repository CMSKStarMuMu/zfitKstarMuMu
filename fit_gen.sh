for i in 0 1 2 3 5 7
do 
	for k in Si Pi
	do
		python fit_gen.py ${i} ${k}
	done
done
