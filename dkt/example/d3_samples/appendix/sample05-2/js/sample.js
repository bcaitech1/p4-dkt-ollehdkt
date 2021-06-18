// 데이터셋은 CSV 파일
d3.csv("mydata.csv",
	function(row){	// 전처리 수행
		return {
			vendor : row.vendor,	// 운영체제 이름은 그대로
			volume : parseInt(row.volume)	// 문자열에서 숫자로 변환
		}
	},
	function(error, dataSet){
		var svgWidth = 600;	// SVG 요소의 넓이
		var svgHeight = 360;	// SVG 요소의 높이
		// 그래프에서 사용할 변수
		var offsetX = 30;	// X 좌표의 오프셋(어긋난 정도)
		var offsetY = 20;	// Y 좌표의 오프셋(어긋난 정도)
		var barElements;	// 막대그래프의 막대 요소를 저장할 변수
		var dataMax = d3.max(	// 데이터의 최댓값을 구함
			dataSet.map(function(d, i){
				return d.volume;
			})
		)
		dataMax = dataMax * 1.2;	// 20% 곱한 값을 최댓값으로 함
		var barWidth = 25;	// 막대의 넓이
		var barMargin = 30;	// 막대 간격
		// 눈금을 표시하고자 축척 설정
		var xScale = d3.scale.ordinal()  // 서로 떨어진 축척 설정
		  .domain(
				dataSet.map(function(d, i){ return d.vendor; })  // 운영체제 설정
			) 
		  .rangeRoundBands([0, svgWidth], 0.5)	// 넓이 지정
		// 눈금을 표시하고자 축척 설정
		var yScale = d3.scale.linear()  // 축척 설정
		  .domain([0, dataMax])	// 출하량 설정
		  .range([svgHeight, 0]) // 실제 출력 크기
		// 그래프 그리기
		barElements = d3.select("#myGraph")
		  .selectAll("rect")	// rect 요소 지정
		  .data(dataSet)	// 데이터를 요소에 연결
		// 데이터 추가
		barElements.enter()	// 데이터 값만큼 반복
		  .append("rect")	// 데이터 수만큼 rect 요소가 추가됨
		  .attr("class", "bar")	// CSS 클래스를 지정
		  .attr("width", xScale.rangeBand())	// 가로 넓이 지정
		  .attr("height", function(d,i){	// 세로 넓이 지정. 2번째 파라미터에 함수를 지정
				return svgHeight - yScale(d.volume);	// 축척 계산
			})
		  .attr("x", function(d, i){	// X 좌표 지정
				return xScale(d.vendor) + offsetX;
			})
		  .attr("y", function(d, i){	// Y 좌표 지정
				return yScale(d.volume) - offsetY;
			})
		// 가로 방향의 눈금을 설정하고 표시
		var xAxisElements = d3.select("#myGraph")
		  .append("g")
		  .attr("class", "axis")
		  .attr("transform", "translate(" + offsetX + ", " + (svgHeight - offsetY) + ")")
		  .call(
				d3.svg.axis()
			  .scale(xScale)  //축척을 적용
			  .orient("bottom") //눈금 표시 위치를 왼쪽으로 지정
			)
		// 세로 방향 눈금을 설정하고 표시
		var yAxisElements = d3.select("#myGraph")
		  .append("g")
		  .attr("class", "axis")
		  .attr("transform", "translate(" + offsetX + ", -" + offsetY + ")")
		  .call(
				d3.svg.axis()
			  .scale(yScale)  //축척을 적용
			  .orient("left") //눈금 표시 위치를 왼쪽으로 지정
			)
	// 체크박스를 클릭했을 때의 정렬 처리
	d3.select("#dataSort").on("change", function(){
		if(this.checked){	// 체크하면 오름차순, 그렇지 않으면 내림차순으로 정렬
			var s = dataSet.sort(function(a, b) { return b.volume - a.volume; });
		}else{
			var s = dataSet.sort(function(a, b) { return a.volume - b.volume; });
		}
		var sortResult = s.map(function(d, i){ return d.vendor; });	// 운영업체별로 변경된 배열을 반환
		xScale.domain(sortResult);	// 새로운 도메인(정렬 후 운영업체 이름을 넣은 배열)을 설정
		// 가로 눈금을 변경
		xAxisElements
		  .transition()
		  .call(
				d3.svg.axis()
			  .scale(xScale)  //축척을 적용
			  .orient("bottom") //눈금 표시 위치를 왼쪽으로 지정
			)
		// 세로 막대를 변경
		barElements
		  .transition()
		  .attr("x", function(d, i){	// X 좌표 지정
				return xScale(d.vendor) + offsetX;
			})
	})
})

