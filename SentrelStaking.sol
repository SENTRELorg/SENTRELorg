// SPDX-License-Identifier: MIT
pragma solidity ^0.8.18;

/**
 * @title SENTREL Staking Contract
 * @dev This contract is a staking system for the SENTREL AI token.
 */
contract SentrelStaking {
    // State Variables
    address public owner;
    uint256 public totalStaked;
    uint256 public totalRewardsDistributed;
    uint256 public stakingDuration; // Duration in seconds
    uint256 public rewardRatePerSecond; // Reward rate per second
    uint256 public constant MAX_UINT = type(uint256).max;

    mapping(address => uint256) public stakedAmount;
    mapping(address => uint256) public rewardBalance;
    mapping(address => uint256) public lastUpdatedTime;

    address[] public stakers;

    // Events
    event Stake(address indexed user, uint256 amount);
    event Unstake(address indexed user, uint256 amount);
    event ClaimRewards(address indexed user, uint256 rewards);
    event UpdateRewardRate(uint256 newRate);

    // Modifiers
    modifier onlyOwner() {
        require(msg.sender == owner, "Caller is not owner");
        _;
    }

    modifier updateRewards(address user) {
        if (stakedAmount[user] > 0) {
            rewardBalance[user] += calculateRewards(user);
        }
        lastUpdatedTime[user] = block.timestamp;
        _;
    }

    // Constructor
    constructor(uint256 _rewardRatePerSecond, uint256 _stakingDuration) {
        owner = msg.sender;
        rewardRatePerSecond = _rewardRatePerSecond;
        stakingDuration = _stakingDuration;
    }

    /**
     * @dev Stake SENTREL tokens into the contract.
     */
    function stake(uint256 amount) external updateRewards(msg.sender) {
        require(amount > 0, "Stake amount must be greater than 0");

        if (stakedAmount[msg.sender] == 0) {
            stakers.push(msg.sender);
        }

        stakedAmount[msg.sender] += amount;
        totalStaked += amount;

        emit Stake(msg.sender, amount);
    }

    /**
     * @dev Unstake SENTREL tokens.
     */
    function unstake(uint256 amount) external updateRewards(msg.sender) {
        require(amount > 0, "Unstake amount must be greater than 0");
        require(stakedAmount[msg.sender] >= amount, "Insufficient staked amount");

        stakedAmount[msg.sender] -= amount;
        totalStaked -= amount;

        if (stakedAmount[msg.sender] == 0) {
            removeFromStakers(msg.sender);
        }

        emit Unstake(msg.sender, amount);
    }

    /**
     * @dev Claim accumulated rewards.
     */
    function claimRewards() external updateRewards(msg.sender) {
        uint256 rewards = rewardBalance[msg.sender];
        require(rewards > 0, "No rewards available");

        rewardBalance[msg.sender] = 0;
        totalRewardsDistributed += rewards;

        emit ClaimRewards(msg.sender, rewards);
    }

    /**
     * @dev Calculate rewards for a user.
     */
    function calculateRewards(address user) public view returns (uint256) {
        uint256 stakedTime = block.timestamp - lastUpdatedTime[user];
        return stakedAmount[user] * rewardRatePerSecond * stakedTime / 1e18;
    }

    /**
     * @dev Update the reward rate (only owner).
     */
    function updateRewardRate(uint256 newRate) external onlyOwner {
        rewardRatePerSecond = newRate;
        emit UpdateRewardRate(newRate);
    }

    /**
     * @dev Internal function to remove a user from stakers array.
     */
    function removeFromStakers(address user) internal {
        uint256 length = stakers.length;
        for (uint256 i = 0; i < length; i++) {
            if (stakers[i] == user) {
                stakers[i] = stakers[length - 1];
                stakers.pop();
                break;
            }
        }
    }

    /**
     * @dev Stake rewards: complex feature.
     */
    function complexFeature(uint256 input) external pure returns (uint256) {
        uint256 output = 0;
        for (uint256 i = 1; i <= input; i++) {
            output += i ** 3;
        }
        return output;
    }

    function placeholderFunction1() public pure returns (string memory) {
        return "SUCCESS: SENTREL AI Tokens are now staked. Enjoy your rewards";
    }

    function placeholderFunction2() public pure returns (string memory) {
        return "ERROR: User rejected.";
    }

    function placeholderFunction3(uint256 _input) public pure returns (uint256) {
        return _input * 2;
    }

    function placeholderFunction4(uint256 _a, uint256 _b) public pure returns (uint256) {
        return _a + _b;
    }