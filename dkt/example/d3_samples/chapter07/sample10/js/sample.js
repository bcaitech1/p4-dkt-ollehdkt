// ●는 추가, 갱신한 곳
// 데이터셋은 CSV 파일●↓
d3.csv("mydata.csv", function(error, data){
	var dataSet = [ ];	// 데이터를 저장할 배열 변수
	var labelName = [ ];	// 레이블을 넣을 배열 변수
	for(var i in data[0]){	// 최초의 데이터만 처리
		dataSet.push(data[0][i]);	// 가로 한 줄 모두를 한꺼번에 넣음
		labelName.push(i);	// 레이블 넣음
	}
	// SVG요소의 넓이와 높이를 구함
	var svgEle = document.getElementById("myGraph");
	var svgWidth = window.getComputedStyle(svgEle, null).getPropertyValue("width");
	var svgHeight = window.getComputedStyle(svgEle, null).getPropertyValue("height");
	svgWidth = parseFloat(svgWidth);	// 값은 단위가 있으므로 단위를 삭제
	svgHeight = parseFloat(svgHeight);	// 값은 단위가 있으므로 단위를 삭제
	// 그래프에서 사용할 변수●↑
	var offsetX = 30;	// X 좌표의 오프셋(어긋남의 정도)
	var offsetY = 20;	// Y 좌표의 오프셋(어긋남의 정도)
	var barElements;	// 막대그래프의 막대 요소를 저장할 변수
	var dataMax = 300;	// 데이터의 최댓값
	var barWidth = 20;	// 막대의 넓이
	var barMargin = 5;	// 막대의 옆 간격
	// 그래프 그리기
	barElements = d3.select("#myGraph")
	  .selectAll("rect")	// rect 요소를 지정
	  .data(dataSet)	// 데이터를 요소에 연결
	// 데이터 추가
	barElements.enter()	// 데이터 수만큼 반복
	  .append("rect")	// 데이터 수만큼 rect 요소가 추가됨
	  .attr("class", "bar")	// CSS 클래스 설정
	  .attr("height", 0)	// 초깃값을 0으로 설정
	  .attr("width", barWidth)	// 넓이 지정
	  .attr("x", function(d, i){
			return i * (barWidth+barMargin)+offsetX;		// X 좌표를 표시 순서×25+offsetX로 함 
		})
	  .attr("y", svgHeight - offsetY)	// 그래프 가장 아래에 좌표를 설정
		// 이벤트 추가
	  .on("mouseover", function(){
			d3.select(this)
			  .style("fill", "red")	// 막대의 칠하기 스타일을 빨간색으로
			})
	  .on("mouseout", function(){
			d3.select(this)
			  .style("fill", "orange")	// 막대의 칠하기 스타일을 오렌지색으로
			})
		// 애니메이션 처리
	  .transition()
	  .duration(1000)	// 1초동안 애니메이션 처리
	  .delay(function(d, i){
			return i * 100;	// 0.1초 대기
		})
	  .attr("y", function(d, i){	// Y 좌표를 지정
			return svgHeight - d - offsetY;	// Y 좌표를 계산
		})
	  .attr("height", function(d,i){	// 넓이 설정. 2번째의 파라미터에 함수를 지정
			return d;	// 데이터 값을 그대로 높이로 지정
		})
	barElements.enter()	// text 요소 지정
	  .append("text")	// text 요소 추가
	  .attr("class", "barNum")	// CSS 클래스 설정
	  .attr("x", function(d, i){	// X 좌표를 지정
			return i * (barWidth+barMargin) + 10+offsetX;	// 막대그래프의 표시 간격을 맞춤
		})
	  .attr("y", svgHeight - 5-offsetY)	// Y 좌표를 지정
	  .text(function(d, i){	// 데이터 표시
			return d;
		})
	// 눈금을 표시하기 위한 스케일 설정
	var yScale = d3.scale.linear()  // 스케일 설정
	  .domain([0, dataMax])   // 원래 크기
	  .range([dataMax, 0]) // 실체 출력 크기
	// 세로 방향의 눈금을 설정하고 표시
	d3.select("#myGraph")
	  .append("g")
	  .attr("class", "axis")
	  .attr("transform", "translate("+offsetX+", "+((svgHeight-300)-offsetY)+")")
	  .call(
			d3.svg.axis()
		  .scale(yScale)  //스케일 적용
		  .orient("left") //눈금의 표시 위치를 왼쪽으로 지정
		)
	// 세로 방향의 선을 표시
	d3.select("#myGraph")
	  .append("rect")
	  .attr("class", "axis_x")
	  .attr("width", svgWidth)
	  .attr("height", 1)
	  .attr("transform", "translate("+offsetX+", "+(svgHeight-offsetY)+")")
	// 막대의 레이블을 표시
	barElements.enter()
	  .append("text")
	  .attr("class", "barName")
	  .attr("x", function(d, i){	// X 좌표를 지정
			return i * (barWidth+barMargin) + 10+offsetX;	// 막대그래프의 표시 간격을 맞춤
		})
	  .attr("y", svgHeight-offsetY+15)
	  .text(function(d, i){
			return labelName[i];	// 레이블 이름을 반환●
		})
});
