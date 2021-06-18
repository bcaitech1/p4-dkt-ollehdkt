// 요소에 데이터를 연결
d3.select("#myGraph")
	.selectAll("rect")	// rect 요소 지정
	.data(["S","B","C","r"])	// 데이터를 요소에 연결
	.enter()	// 데이터 개수만큼 반복
	.append("rect")	// 데이터의 개수만큼 rect 요소가 추가됨
	.datum(function(d, i){
		console.log(i+" = "+d);	// 순서와 데이터 내용 출력
		return d;	// 설정할 데이터를 반환
	})
