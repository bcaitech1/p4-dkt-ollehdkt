d3.csv("mydata.csv", function(error, data){
	var dataSet = [ ];	// 데이터를 저장할 배열 변수
	var labelName = [ ];	// 레이블을 넣을 배열 변수
	for(var i in data[0]){	// 레이블을 처리
		labelName.push(i);	// 레이블 넣음
	}
	for(var j=0; j<data.length-1; j++){	// 데이터 수만큼 반복
		for(var i in data[j]){	// 데이터를 처리
			dataSet.push(data[j][i]);	// 가로 한 줄 모두를 한꺼번에 넣음
		}
	}
	console.log(dataSet);
	console.log("-----");
	console.log(labelName);
});
