var svgHeight = 240;	// SVG 요소의 높이
var offsetX = 30;	// X 좌표의 오프셋(어긋남의 정도)
var offsetY = 20;	// Y 좌표의 오프셋(어긋남의 정도)
var barElements;	// 막대그래프의 막대 요소를 저장할 변수
var dataSet = [120, 70, 175, 80, 220];
// 그래프 그리기
barElements = d3.select("#myGraph")
	.selectAll("rect")	// rect 요소를 지정
	.data(dataSet)	// 데이터를 요소에 연결
// 데이터 추가
barElements.enter()	// 데이터 수만큼 반복
	.append("rect")	// 데이터 수만큼 rect 요소가 추가됨
	.attr("class", "bar")	// CSS 클래스 설정
	.attr("height", function(d,i){	// 넓이 설정. 2번째의 파라미터에 함수를 지정
		return d;	// 데이터 값을 그대로 높이로 지정
	})
	.attr("width", 20)	// 넓이 지정
	.attr("x", function(d, i){
		return i * 25+offsetX;		// X 좌표를 표시 순서×25+offsetX로 함 
	})
	.attr("y", function(d, i){	// Y 좌표를 지정
		return svgHeight - d - offsetY;	// Y 좌표를 계산
	})
barElements.enter()	// text 요소 지정
	.append("text")	// text 요소 추가
	.attr("class", "barNum")	// CSS 클래스 설정
	.attr("x", function(d, i){	// X 좌표를 지정
		return i * 25 + 10+offsetX;	// 막대그래프의 표시 간격을 맞춤
	})
	.attr("y", svgHeight - 5-offsetY)	// Y 좌표를 지정
	.text(function(d, i){	// 데이터 표시
		return d;
	})
// 눈금을 표시하기 위한 스케일 설정
var yScale = d3.scale.linear()  // 스케일 설정
	.domain([0, 300])   // 원래 크기
	.range([300, 0]) // 실체 출력 크기
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
// 가로 방향의 선을 표시●↓
d3.select("#myGraph")
	.append("rect")
	.attr("class", "axis_x")
	.attr("width", 320)
	.attr("height", 1)
	.attr("transform", "translate("+offsetX+", "+(svgHeight-offsetY)+")")
// 막대의 레이블을 표시
barElements.enter()
	.append("text")
	.attr("class", "barName")
	.attr("x", function(d, i){	// X 좌표를 지정
		return i * 25 + 10+offsetX;	// 막대그래프의 표시 간격을 맞춤
	})
	.attr("y", svgHeight-offsetY+15)
	.text(function(d, i){
		return ["A", "B", "C", "D", "E"][i];	// 레이블 이름을 반환
	})