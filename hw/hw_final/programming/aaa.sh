for dir in ~/benchmarksuite/*/
do
  echo ${dir}
  cd ${dir}
  rm b.txt result.txt
#  ./fp ${dir}a.txt ${dir}x.txt
done

