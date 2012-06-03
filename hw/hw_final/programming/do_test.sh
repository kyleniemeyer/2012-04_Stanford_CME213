for dir in ~/benchmarksuite/*/
do
  echo ${dir}
  ./fp ${dir}a.txt ${dir}x.txt
done

