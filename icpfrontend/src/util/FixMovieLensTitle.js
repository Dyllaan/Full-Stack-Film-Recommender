//Movielens has titles with "Lion King, The" instead of "The Lion King"
export default function fixMovieLensTitle(str) {
    let newStr = str;
    const ending = ", The";
    if (newStr.endsWith(ending)) {
        newStr = newStr.slice(0, -ending.length);
        newStr = 'The ' + newStr;
    }

    if(newStr.includes("Â")) {
        newStr = newStr.replace("Â", "");
    }
    return newStr;
}