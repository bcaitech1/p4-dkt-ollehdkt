d3.select("#myBar")	// ID 이름을 myBar로 지정
	.attr("x", "10px")	// X 좌표를 10px로 설정
	.attr("y", "50px")	// Y 좌표를 50px로 설정
	.attr("width", "200px")	// 넓이를 200px로 설정
	.attr("height", "30px")	// 높이를 30px로 설정
	.transition()	// 애니메이션 설정
	.duration(3000)	// 3초간 변화
	.attr("width", "50px")	// 넓이를 50px로 설정
