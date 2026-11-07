import { jsx as _jsx } from "react/jsx-runtime";
import { TEXT_INPUT_CLASS } from '../../constants/styles';
export default function TextInput({ className = '', ...props }) {
    return (_jsx("input", { ...props, className: `${TEXT_INPUT_CLASS} outline-none w-full ${className}` }));
}
