'''
author: @PenguinsResearch © 2022
'''

import re 

s1 = u'ÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚÝàáâãèéêìíòóôõùúýĂăĐđĨĩŨũƠơƯưẠạẢảẤấẦầẨẩẪẫẬậẮắẰằẲẳẴẵẶặẸẹẺẻẼẽẾếỀềỂểỄễỆệỈỉỊịỌọỎỏỐốỒồỔổỖỗỘộỚớỜờỞởỠỡỢợỤụỦủỨứỪừỬửỮữỰựỲỳÝýỴỵỶỷỸỹ'
s0 = u'AAAAEEEIIOOOOUUYaaaaeeeiioooouuyAaDdIiUuOoUuAaAaAaAaAaAaAaAaAaAaAaAaEeEeEeEeEeEeEeEeIiIiOoOoOoOoOoOoOoOoOoOoOoOoUuUuUuUuUuUuUuYyYyYyYyYy'


def norm_text(input_str):
	# remove special characters
	input_str = re.sub(f'[^a-zA-Z0-9().{s1} ]', '', input_str)
	return input_str

def remove_accents(input_str):
	s = ''
	for c in input_str:
		if c in s1:
			s += s0[s1.index(c)]
		else:
			s += c
	return s


if __name__=="__main__":
    print(remove_accents("Đây là một câu hỏi tiếng Việt có dấu"))
    print(norm_text("Đây là một câu hỏi tiếng Việt có dấu và ký tự đặc biệt !@#$%^&*()_+"))
