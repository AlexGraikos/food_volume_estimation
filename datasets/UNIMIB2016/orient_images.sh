# Orient images in correct TopLeft format to align with polygon masks
for img_file in ./*.jpg
do 
    convert "$img_file" -orient TopLeft "$img_file" ; 
done


