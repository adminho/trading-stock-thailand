# reference: https://iapi.bot.or.th/Developer?lang=th
# 17/07/2017
""""
บริการข้อมูล BOT API

อัตราแลกเปลี่ยน (8 APIs)
1) อัตราแลกเปลี่ยนถัวเฉลี่ยถ่วงน้ำหนักระหว่างธนาคาร (รายวัน)
2) อัตราแลกเปลี่ยนถัวเฉลี่ยถ่วงน้ำหนักระหว่างธนาคาร (รายเดือน)
3) อัตราแลกเปลี่ยนถัวเฉลี่ยถ่วงน้ำหนักระหว่างธนาคาร (รายไตรมาส)
4) อัตราแลกเปลี่ยนถัวเฉลี่ยถ่วงน้ำหนักระหว่างธนาคาร (รายปี)
5) อัตราแลกเปลี่ยนเฉลี่ย (รายวัน)
6) อัตราแลกเปลี่ยนเฉลี่ย (รายเดือน)
7) อัตราแลกเปลี่ยนเฉลี่ย (รายไตรมาส)
8) อัตราแลกเปลี่ยนเฉลี่ย (รายปี)

อัตราดอกเบี้ย (12 APIs)
1) อัตราดอกเบี้ยต่างประเทศ (อัตราร้อยละต่อปี)
2) Thai Baht Implied Interest Rates (Percent per annum)
3) Spot Rate (ดอลลาร์/บาท)
4) Swap point ตลาดในประเทศ (สตางค์)
5) อัตราดอกเบี้ยการกู้ยืมระหว่างธนาคาร (อัตราร้อยละต่อปี)
6) อัตราดอกเบี้ยนโยบาย (อัตราร้อยละต่อปี)
7) อัตราดอกเบี้ยอ้างอิงระยะสั้นตลาดกรุงเทพรายธนาคาร (อัตราร้อยละต่อปี)
8) อัตราดอกเบี้ยอ้างอิงระยะสั้นตลาดกรุงเทพเฉลี่ย (อัตราร้อยละต่อปี)
9) อัตราดอกเบี้ยเงินฝากสำหรับบุคคลธรรมดาของธนาคารพาณิชย์ (อัตราร้อยละต่อปี)
10) อัตราดอกเบี้ยเงินฝากต่ำสุด-สูงสุดสำหรับบุคคลธรรมดาของธนาคารพาณิชย์ (อัตราร้อยละต่อปี)
11) อัตราดอกเบี้ยเงินให้สินเชื่อของธนาคารพาณิชย์ (อัตราร้อยละต่อปี)
12) อัตราดอกเบี้ยเงินให้สินเชื่อเฉลี่ยของธนาคารพาณิชย์ (อัตราร้อยละต่อปี)

ผลการประมูลตราสารหนี้ (1 API)
1) ผลการประมูลตราสารหนี้

"""

#Reference: https://iapi.bot.or.th/Developer/Home/Api/9
"""
EXCHANGE_RATE
Exchange rate is the price of a currency with respect to another currency. For instance, The value in Thai Baht of 1 US dollar will equal 40 Baht. In general, there are 2 major exchange rate regimes; namely, fixed and floating exchange rate regimes. At present, Thailand adopts a floating exchange rate regime.

1) API - Daily Weighted-average Interbank Exchange Rate - THB / USD
2) API - Monthly Weighted-average Interbank Exchange Rate - THB / USD
3) API - Quarterly Weighted-average Interbank Exchange Rate - THB / USD
4) API - Annual Weighted-average Interbank Exchange Rate - THB / USD
5) API - Daily Average Exchange Rate - THB / Foreign Currency
6) API - Monthly Average Exchange Rate - THB / Foreign Currency
7) API - Quarterly Average Exchange Rate - THB / Foreign Currency
8) API - Annual Average Exchange Rate - THB / Foreign Currency

-----------------------
INTEREST RATE
Interest Rate covers domestic money market interest rates, namely the Policy Rate, Interbank Transaction Rates, Bangkok Interbank Offered Rate (BIBOR), Thai Baht Implied Interest Rate, End-of-day Liquidity Rate, Deposit Interest Rates and Loan Interest Rates of Commercial banks, external interest rates (US interest rates, LIBORs, and SIBOR) and other rates such as spot rate and swap point.

1) API - External Interest Rates (Percent per annum)
2) API - Thai Baht Implied Interest Rates (Percent per annum)
3) API - Spot Rate USD/THB
4) API - Swap point - Onshore (in Satangs)
5) API - Interbank Transaction Rates (Percent per annum)
6) API - Policy Rate (Percent per annum)
7) API - Bangkok Interbank Offered Rate (BIBOR) by Banks (Percent per annum)
8) API - Average Bangkok Interbank Offered Rate (BIBOR) (Percent per annum)
9) API - Deposit Interest Rates for Individuals of Commercial Banks (Percent per annum)
10) API - Min-Max Deposit Interest Rates for Individuals of Commercial Banks (Percent per annum)
11) API - Loan Interest Rates of Commercial Banks (Percent per annum)
12) API - Average Loan Interest Rates of Commercial Banks (Percent per annum)

-----------------------
DEBT SECURITIES AUCTION
Debt securities auction is a method employed by the issuers to sell the securities in the primary market normally aimed at institutional investors as they have large amount of fund and investment expertise. The auction provides several advantages to the issuers. Among these include the relatively low cost of borrowing, less time spent on selling process, and, under normal market conditions, the offering amount is usually fully subscribed. Thus the issuer can manage their debt effectively.

1) API - Debt Securities Auction Result

"""
import requests
import json  
from time import gmtime, strftime
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# For Exchange Rate only
def _requestExchangeRate(url, start_date, end_date):
	# url = "https://iapi.bot.or.th/Stat/Stat-ReferenceRate/DAILY_REF_RATE_V1/"
	# querystring = {"start_period":"2017-01-12","end_period":"2017-01-15"}
	querystring = {"start_period":start_date,"end_period":end_date}
	headers = {
		'api-key': "U9G1L457H6DCugT7VmBaEacbHV9RX0PySO05cYaGsm"
	}
	response = requests.request("GET", url, headers=headers, params=querystring)
	json_text = response.text
	
	"""
	Response example: 
	{"result":{
			"success":"true",
			"api":"Daily Weighted-average Interbank Exchange Rate - THB / USD",
			"timestamp":"2017-07-20 23:56:06",
			"data":{
					"data_header":{
									"report_name_eng":"Rates of Exchange of Commercial Banks in Bangkok Metropolis (2002-present)",
									"report_name_th":"อัตราแลกเปลี่ยนเฉลี่ยของธนาคารพาณิชย์ในกรุงเทพมหานคร (2545-ปัจจุบัน)",
									"report_uoq_name_eng":"(Unit : Baht / 1 Unit of Foreign Currency)",
									"report_uoq_name_th":"(หน่วย : บาท ต่อ 1 หน่วยเงินตราต่างประเทศ)",
									"report_source_of_data":[{"source_of_data_eng":"Bank of Thailand","source_of_data_th":"ธนาคารแห่งประเทศไทย"}],
									"report_remark":[],
									"last_updated":"2017-07-20"
									},
					"data_detail":[
									{"period":"2002-01-15","rate":"43.9200000"},
									{"period":"2002-01-14","rate":"43.9230000"}]
					}
			}
	}
	"""
	json_obj = json.loads(json_text)  
	success = json_obj["result"]["success"]
	if success == "false":
		error = json_obj["result"]["error"]
		"""
		Example when error occur
		{'result': {'success': 'false', 'api': 'Monthly Weighted-average Interbank Exchange Rate - THB / USD', 
					'timestamp': '2017-07-21 13:38:41', 
					'error': [{'code': 'es003', 'message': 'Parameter: start_period format must be yyyy-mm'}, 
								{'code': 'es003', 'message': 'Parameter: end_period format must be yyyy-mm'}]
					}
		}
		"""		
		raise ValueError(error)	
			
	data_detail = json_obj["result"]["data"]["data_detail"]
	return data_detail
	# end function

def _df_DAILY_REF_RATE(data_detail):
	date_list, rate_list = [], []
	for value in data_detail:			
		date_list.append(value["period"])
		rate_list.append(value["rate"])
		
	# convert string to date
	date_list = [ datetime.datetime.strptime(ts, "%Y-%m-%d") for ts in date_list]
	# convert string to float
	rate_list = [float(i.strip()) for i in rate_list]
	# create dataframe for collecting all detail
	df = pd.DataFrame(data = rate_list, index=date_list, columns =["DAILY_RATE"] )	
	return df

def _df_DAILY_AVG_EXG_RATE(data_detail):
	date_list, rate_list, currencyname_list = [], [], []
	for value in data_detail:		
		date_list.append(value["period"])
		rate_list.append(value["mid_rate"])
		currencyname_list.append(value["currency_name_eng"])

	templist = []
	# remove duplicated currency_name_eng (and remove whitespace on the right and left side)	
	[templist.append(name.strip()) for name in currencyname_list if name.strip() not in templist]
	currencyname_list = templist
		
	num_column = len(currencyname_list)	
	num_row = int(len(date_list)/num_column)
	assert num_row == int(len(rate_list)/num_column)
	
	# reshape 
	date_list = np.reshape(date_list, (num_row, num_column) )
	date_list = date_list[0:, 0]
	
	# convert string to float
	rate_list = [float(i.strip()) for i in rate_list]
	# reshape 
	rate_list = np.reshape(rate_list, (num_row, num_column))	
		
	# convert string to date
	date_list = [ datetime.datetime.strptime(ts, "%Y-%m-%d") for ts in date_list]
	# create dataframe for collecting all detail
	df = pd.DataFrame(data = rate_list, index=date_list, columns = currencyname_list )	
	return df

"""
วางแผนว่าจะทำคล้ายๆ  _df_DAILY_AVG_EXG_RATE เนื่องจากมันมี interest rate หลายประเภท
def _df_THB_IMPL_INT_RATE(data_detail):
	date_list, interest_list = [], []
	for value in data_detail:			
		date_list.append(value["period"])
		interest_list.append(value["interest_rate"])
		
	# convert string to date
	date_list = [ datetime.datetime.strptime(ts, "%Y-%m-%d") for ts in date_list]
	# convert string to float
	interest_list = [float(i.strip()) for i in interest_list]	
	# create dataframe for collecting all detail
	df = pd.DataFrame(data = interest_list, index=date_list, columns =["INTEREST_RATE"] )	
	return df
"""

# API must get data less than 31 days per request
def _fetch_all_rate(url, start_date, end_date, createDataFrame):
	period1 = datetime.timedelta(days=1)
	period31 = datetime.timedelta(days=30)
	STOP_DDD = datetime.datetime.strptime(end_date, "%Y-%m-%d").date()
	start_ddd = datetime.datetime.strptime(start_date, "%Y-%m-%d").date()
			
	to_ddd =  start_ddd + period31
	df_list = []
	while(to_ddd < STOP_DDD):
		data_detail = _requestExchangeRate(url, start_ddd, to_ddd)	
		df = createDataFrame(data_detail)
		# save dataframe
		df_list.append(df)
		# shift period31
		start_ddd = to_ddd + period1
		to_ddd =  to_ddd + period31		
	
	# get remain data
	data_detail = _requestExchangeRate(url, start_ddd, STOP_DDD)
	df = createDataFrame(data_detail)
	
	# save dataframe
	df_list.append(df)
	df_result = pd.concat(df_list)
	df_result = df_result.sort_index() # order date	
	return df_result
	
# อัตราแลกเปลี่ยนถัวเฉลี่ยถ่วงน้ำหนักระหว่างธนาคาร (รายวัน)
"""
ครอบคลุมอัตราแลกเปลี่ยนระหว่างเงินบาทเทียบกับเงินสกุลดอลลาร์ สรอ. โดยอัตราแลกเปลี่ยนดังกล่าว
เป็นอัตราซื้อขายทันที และเป็นอัตราเฉลี่ยถ่วงน้ำหนักรายวันจากข้อมูลการซื้อขายเงินตราต่างประเทศ
ระหว่างธนาคารพาณิชย์
"""
# Daily Weighted-average Interbank Exchange Rate - THB / USD
def get_DAILY_REF_RATE(startDate, endDate):	
	return _fetch_all_rate("https://iapi.bot.or.th/Stat/Stat-ReferenceRate/DAILY_REF_RATE_V1/", 
							startDate, endDate, 
							_df_DAILY_REF_RATE)	
	
# อัตราแลกเปลี่ยนถัวเฉลี่ยถ่วงน้ำหนักระหว่างธนาคาร (รายเดือน)
"""
 ครอบคลุมอัตราแลกเปลี่ยนระหว่างเงินบาทเทียบกับเงินสกุลอื่น ๆ รวมทั้งสิ้น 48 สกุล โดย
อัตราแลกเปลี่ยนดังกล่าวเป็นอัตราซื้อขายทันที ซึ่งเป็นอัตราเฉลี่ยรายวันจากข้อมูลธนาคารพาณิชย์บางแห่ง
และบางส่วนเก็บรวบรวมจากอัตราปิดตลาดนิวยอร์ค และวารสาร Financial Times ที่คำนวณ
ผ่านอัตราซื้อขายเงินดอลลาร์ สรอ. ในตลาดกรุงเทพฯ
"""
# Daily Average Exchange Rate - THB / Foreign Currency
def get_DAILY_AVG_EXG_RATE(startDate, endDate):	
	return _fetch_all_rate("https://iapi.bot.or.th/Stat/Stat-ExchangeRate/DAILY_AVG_EXG_RATE_V1/",
							startDate, endDate,
							_df_DAILY_AVG_EXG_RATE)
	

# Thai Baht Implied Interest Rates (Percent per annum)
"""
Thai baht Implied Interest Rates เป็นอัตราดอกเบี้ยของการกู้ยืมเงินบาทผ่านตลาด Swap 
ซึ่งธนาคารแห่งประเทศไทยเผยแพร่ทั้งของตลาดในประเทศ ที่เป็นการกู้ยืมระหว่างธนาคาร 
และการกู้ยืมระหว่างธนาคารพาณิชย์กับลูกค้า และอัตราดอกเบี้ย ฯ ของตลาดต่างประเทศ ที่เป็นการซื้อขายระหว่างธนาคารพาณิชย์ กับ Non-Resident
"""
# Thai Baht Implied Interest Rates (Percent per annum)
"""
def get_THB_IMPL_INT_RATE(startDate, endDate):	
	return _fetch_all_rate("https://iapi.bot.or.th/Stat/Stat-ThaiBahtImpliedInterestRate/THB_IMPL_INT_RATE_V1/",
							startDate, endDate,
							_df_THB_IMPL_INT_RATE)					
"""

# ปัญหาแต่ละ API ต้องการพารามิเตอร์ที่แตกต่างกัน
# ผมเลยเลือกทำ API แค่ 2 ตัวก่อน

startDate = '2017-03-01'
endDate = strftime("%Y-%m-%d", gmtime())    
df1 = get_DAILY_REF_RATE(startDate, endDate) 
df2 = get_DAILY_AVG_EXG_RATE(startDate, endDate) 

# for debug
df1.to_csv("daily_rate_test.csv")
df2.to_csv("daily_avg_exg_rate_test.csv")

df_merge = df1.join(df2, how="left")
df_merge.to_csv("daily_rate_merge_test.csv")
print(df_merge.head())

# plot graph
plt.plot( df_merge.index, df_merge['DAILY_RATE'].values, )
plt.plot( df_merge.index, df_merge['USA : DOLLAR (USD)'].values)
plt.legend(['DAILY_RATE', 'USA : DOLLAR (USD)'])

plt.xlabel("Date")
plt.ylabel("Exchange rate")
plt.show()

"""
df3 = get_THB_IMPL_INT_RATE(startDate, endDate)
df3.to_csv("interest_rate_test.csv")
df3.plot()
plt.show()"""
