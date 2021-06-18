// 요소에 데이터를 연결
d3.select("#myGraph")
	.selectAll("rect")	// rect 요소 지정
	.data([0,0,0,0,0])	// 더미 데이터를 요소에 연결
	.enter()	// 데이터 개수만큼 반복
	.append("rect")	// 데이터의 개수만큼 rect 요소가 추가됨
	.datum(function(){	// 모든 요소에 난수값을 연결
		return Math.random();	// 난수 값을 반환
	})
