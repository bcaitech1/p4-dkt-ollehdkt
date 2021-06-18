d3.selectAll(".bar")	// CSS 클래스 이름에 bar를 지정
	.style("fill", function(d,i){	// 2번째의 파라미터에 함수를 지정
		if(i == 2){	// 순서를 조사
			return "red";	// 3번째라면 빨간색을 나타내는 문자를 반환
		}	
	})
