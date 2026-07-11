def counting(s):
    count=0;
    count1=0;
    for x in s:
        if x in "aeiouAEIOU":
            count+=1;
        else:
            count1+=1;
    return count,count1;

input1=input("Enter a string: ");
ans,ans1=counting(input1);
print("Number of vowels in input string: ",ans,"Number of consonants in input string: ",ans1);